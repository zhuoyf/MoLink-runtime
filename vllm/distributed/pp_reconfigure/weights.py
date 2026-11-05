# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import AutoWeightsLoader


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
def load_local_layers(config, layers, device):
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
                        yield name, reader.get_tensor(name)
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
    }
