# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import AutoWeightsLoader


class CPUWeightCache:
    """Bounded retained CPU tensors; misses retain the existing disk path."""

    def __init__(self, budget_bytes=0):
        if budget_bytes < 0:
            raise ValueError("CPU weight cache budget must be nonnegative")
        self.budget_bytes = budget_bytes
        self.tensors = OrderedDict()
        self.resident_bytes = self.hits_bytes = self.miss_bytes = 0
        self.evicted_bytes = 0
        self.lock = threading.Lock()
        self.thread = None
        self.prefetch_error = None
        self.prefetch_started = self.prefetch_finished = None

    def _put(self, name, tensor):
        size = tensor.numel() * tensor.element_size()
        if not self.budget_bytes or size > self.budget_bytes:
            return tensor
        with self.lock:
            if name in self.tensors:
                return self.tensors[name]
            while self.resident_bytes + size > self.budget_bytes:
                _, old = self.tensors.popitem(last=False)
                released = old.numel() * old.element_size()
                self.resident_bytes -= released
                self.evicted_bytes += released
                del old
            cached = tensor.clone()
            self.tensors[name] = cached
            self.resident_bytes += size
            return cached

    def get(self, name, loader):
        with self.lock:
            if name in self.tensors:
                tensor = self.tensors[name]
                self.tensors.move_to_end(name)
                self.hits_bytes += tensor.numel() * tensor.element_size()
                return tensor
        tensor = loader()
        with self.lock:
            self.miss_bytes += tensor.numel() * tensor.element_size()
        return self._put(name, tensor)

    def prefetch(self, config, layers):
        if not self.budget_bytes or (
            self.thread is not None and self.thread.is_alive()
        ):
            return
        prefixes = tuple(f"model.layers.{i}." for i in layers)

        @torch.inference_mode()
        def run():
            self.prefetch_started = time.time()
            try:
                selected = 0
                for filename in local_weight_files(config):
                    with safe_open(filename, framework="pt", device="cpu") as reader:
                        for name in sorted(reader.keys()):
                            if name.startswith(prefixes):
                                tensor = reader.get_tensor(name)
                                selected += tensor.numel() * tensor.element_size()
                                if selected > self.budget_bytes:
                                    return
                                self._put(name, tensor)
            except Exception as exc:
                # Prefetch is opportunistic; the loader reports real disk failures.
                self.prefetch_error = f"{type(exc).__name__}: {exc}"
            finally:
                self.prefetch_finished = time.time()

        self.prefetch_error = None
        self.prefetch_finished = None
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def metrics(self):
        with self.lock:
            return dict(
                budget_bytes=self.budget_bytes,
                resident_bytes=self.resident_bytes,
                hits_bytes=self.hits_bytes,
                miss_bytes=self.miss_bytes,
                evicted_bytes=self.evicted_bytes,
                prefetch_error=self.prefetch_error,
                prefetch_started=self.prefetch_started,
                prefetch_finished=self.prefetch_finished,
            )


def local_weight_files(config):
    checkpoint = Path(config.model_config.model)
    if not checkpoint.is_dir():
        raise ValueError(f"local model directory is required: {checkpoint}")
    _, files, use_safetensors = DefaultModelLoader(config.load_config)._prepare_weights(
        str(checkpoint), None, config.model_config.revision, False, None
    )
    if not use_safetensors:
        raise ValueError("local layer loading requires a safetensors checkpoint")
    return files


@torch.inference_mode()
def load_local_layers(config, layers, device, weight_cache=None):
    """Populate only the staged layers using vLLM's native HF weight mapper."""
    staged = nn.Module()
    decoder = nn.Module()
    decoder.add_module("layers", nn.ModuleDict({str(i): m for i, m in layers.items()}))
    staged.add_module("model", decoder)
    prefixes = tuple(f"model.layers.{i}." for i in layers)
    expected: set[str] = set()
    for name, _ in staged.named_parameters():
        if ".qkv_proj." in name:
            expected.update(
                name.replace(".qkv_proj.", f".{part}_proj.") for part in ("q", "k", "v")
            )
        elif ".gate_up_proj." in name:
            expected.update(
                name.replace(".gate_up_proj.", f".{part}_proj.")
                for part in ("gate", "up")
            )
        else:
            expected.add(name)

    def weights():
        seen = set()
        for filename in local_weight_files(config):
            with safe_open(filename, framework="pt", device="cpu") as reader:
                for name in sorted(reader.keys()):
                    if name.startswith(prefixes):
                        seen.add(name)
                        tensor = (
                            weight_cache.get(
                                name, lambda name=name: reader.get_tensor(name)
                            )
                            if weight_cache is not None
                            else reader.get_tensor(name)
                        )
                        yield name, tensor
        if missing := expected - seen:
            raise ValueError(f"checkpoint is missing layer weights: {sorted(missing)}")

    loaded = AutoWeightsLoader(staged).load_weights(
        weights(), mapper=LlamaModel.hf_to_vllm_mapper
    )
    if missing := {name for name, _ in staged.named_parameters()} - loaded:
        raise ValueError(f"layer parameters were not loaded: {sorted(missing)}")
    process_weights_after_loading(staged, config.model_config, device)
    return {
        "loaded_bytes": sum(p.numel() * p.element_size() for p in staged.parameters()),
        "layers": sorted(layers),
        "checkpoint": config.model_config.model,
        "cpu_cache": weight_cache.metrics() if weight_cache is not None else None,
    }
