# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.pp_reconfigure.errors import ReconfigurationValidationError
from vllm.distributed.pp_reconfigure.monitor import write_event
from vllm.distributed.pp_reconfigure.transport import NixlTransport, PPTransport
from vllm.distributed.pp_reconfigure.weights import (
    CPUWeightCache,
    load_local_layers,
    local_weight_files,
)
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.worker.utils import allocate_kv_cache, bind_kv_cache


def layer_range(stage: int, boundary: int, total: int) -> tuple[int, int]:
    if type(boundary) is not int or not 0 < boundary < total:
        raise ReconfigurationValidationError(
            f"boundary must be an integer in [1, {total - 1}]"
        )
    return (0, boundary) if stage == 0 else (boundary, total)


@dataclass
class Prepared:
    boundary: int
    old_boundary: int
    old_context: dict
    new_context: dict
    old_layers: list
    new_layers: dict
    old_kv: dict
    new_kv: dict
    old_config: Any
    new_config: Any
    moving: list[int]
    source: int
    transport: Any
    weights_ready: bool = False
    kv_ready: bool = False
    committed: bool = False


class LayerMigration:
    def __init__(self, worker):
        self.worker = worker
        self.runner = worker.model_runner
        self.config = worker.vllm_config
        self.stage = get_pp_group().rank_in_group
        self.prepared: Prepared | None = None
        self.weight_cache = CPUWeightCache(envs.VLLM_PP_RECONFIGURE_WEIGHT_CACHE_BYTES)

    @property
    def model(self):
        return self.runner.get_model().model

    def describe(self, boundary):
        config = self.config
        from vllm.model_executor.models.llama import LlamaForCausalLM

        if not envs.VLLM_PP_RECONFIGURE_ENABLED:
            raise ReconfigurationValidationError(
                "set VLLM_PP_RECONFIGURE_ENABLED=1 on every worker"
            )
        parallel = config.parallel_config
        if (
            parallel.pipeline_parallel_size,
            parallel.tensor_parallel_size,
            parallel.data_parallel_size,
        ) != (2, 1, 1):
            raise ReconfigurationValidationError("prototype requires PP=2, TP=1, DP=1")
        if not config.model_config.enforce_eager:
            raise ReconfigurationValidationError("prototype requires --enforce-eager")
        if type(self.runner.get_model()) is not LlamaForCausalLM:
            raise ReconfigurationValidationError("prototype supports LlamaForCausalLM")
        if any(
            (
                config.quant_config,
                config.speculative_config,
                config.lora_config,
                config.kv_transfer_config,
            )
        ):
            raise ReconfigurationValidationError(
                "requires plain weights and local full-attention KV"
            )
        groups = self.runner.kv_cache_config.kv_cache_groups
        if len(groups) != 1 or type(groups[0].kv_cache_spec) is not FullAttentionSpec:
            raise ReconfigurationValidationError(
                "requires one uniform FullAttentionSpec KV group"
            )
        if groups[0].kv_cache_spec.kv_quant_mode != 0:
            raise ReconfigurationValidationError("prototype requires unquantized KV")
        backend = envs.VLLM_PP_RECONFIGURE_KV_TRANSPORT
        if backend not in ("nixl", "pp"):
            raise ReconfigurationValidationError(
                "VLLM_PP_RECONFIGURE_KV_TRANSPORT must be nixl or pp"
            )
        local_weight_files(config)
        start, end = layer_range(
            self.stage, boundary, self.model.config.num_hidden_layers
        )
        old_boundary = (
            self.model.end_layer if self.stage == 0 else self.model.start_layer
        )
        if old_boundary == boundary:
            raise ReconfigurationValidationError("boundary already active")
        return {
            "stage": self.stage,
            "old_boundary": old_boundary,
            "boundary": boundary,
            "start_layer": start,
            "end_layer": end,
            "num_blocks": self.runner.kv_cache_config.num_blocks,
            "spec": repr(groups[0].kv_cache_spec),
            "layout": str(config.cache_config.get_resolved_kv_cache_layout()),
            "transport": backend,
        }

    def prepare(self, boundary):
        description = self.describe(boundary)
        if self.prepared is not None:
            raise ValueError("a migration is already prepared")
        runner = self.runner
        context = runner.compilation_config.static_forward_context
        old_context = dict(context)
        old_config = runner.kv_cache_config
        old_names = old_config.kv_cache_groups[0].layer_names
        old_kv = dict(zip(sorted(old_names, key=_layer_index), runner.kv_caches))
        start, end = description["start_layer"], description["end_layer"]
        names = [f"model.layers.{i}.self_attn.attn" for i in range(start, end)]
        # Keep native layout and exact block capacity; never rebuild the pool.
        from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups

        spec = old_config.kv_cache_groups[0].kv_cache_spec
        override = self.config.cache_config.num_gpu_blocks_override
        try:
            self.config.cache_config.num_gpu_blocks_override = old_config.num_blocks
            new_config = get_kv_cache_config_from_groups(
                self.config,
                [KVCacheGroupSpec(names, spec)],
                old_config.num_blocks * spec.page_size_bytes * len(names),
            )
        finally:
            self.config.cache_config.num_gpu_blocks_override = override
        old_boundary = description["old_boundary"]
        moving = list(range(min(boundary, old_boundary), max(boundary, old_boundary)))
        source = 0 if boundary < old_boundary else 1
        # Retain the old allocation until all ranks have committed or rolled back.
        weight_bytes = 0
        if self.stage != source:
            exemplar = self.model.layers[self.model.start_layer]
            weight_bytes = len(moving) * sum(
                t.numel() * t.element_size() for t in exemplar.state_dict().values()
            )
        free, _ = torch.accelerator.get_memory_info(self.worker.device)
        allocator_free = torch.accelerator.memory_reserved(
            self.worker.device
        ) - torch.accelerator.memory_allocated(self.worker.device)
        required = new_config.kv_cache_tensors[0].size + weight_bytes + (32 << 20)
        if required > free + allocator_free:
            raise ValueError(
                f"insufficient migration headroom: need {required}, "
                f"available {free + allocator_free}; lower --num-gpu-blocks-override"
            )
        new_layers = {}
        try:
            if self.stage != source:
                from vllm.model_executor.models.llama import LlamaDecoderLayer

                with (
                    set_current_vllm_config(self.config),
                    torch.device(self.worker.device),
                    set_default_torch_dtype(self.config.model_config.dtype),
                    torch.inference_mode(),
                ):
                    for index in moving:
                        new_layers[index] = LlamaDecoderLayer(
                            vllm_config=self.config, prefix=f"model.layers.{index}"
                        )
            new_context = {name: context[name] for name in names}
            new_kv = allocate_kv_cache(
                new_config,
                self.worker.device,
                self.config.cache_config.get_resolved_kv_cache_layout(),
                getattr(runner, "kernel_block_sizes", None)
                or runner._kernel_block_sizes,
            )
            for name in names:
                if name in old_kv:
                    new_kv[name].copy_(old_kv[name])
            torch.accelerator.synchronize(self.worker.device)
        finally:
            context.clear()
            context.update(old_context)
        backend = description["transport"]
        transport = NixlTransport() if backend == "nixl" else PPTransport()
        self.prepared = Prepared(
            boundary,
            old_boundary,
            old_context,
            new_context,
            list(self.model.layers),
            new_layers,
            old_kv,
            new_kv,
            old_config,
            new_config,
            moving,
            source,
            transport,
        )
        description["weight_manifest"] = [
            (index, name, list(tensor.shape), str(tensor.dtype))
            for index in moving
            for name, tensor in (
                self.model.layers[index] if self.stage == source else new_layers[index]
            )
            .state_dict()
            .items()
        ]
        return description

    def connect(self):
        self._transaction().transport.connect()
        return {"connected": True}

    def load_weights(self):
        tx = self._transaction()
        result = {"loaded_bytes": 0, "layers": []}
        if self.stage != tx.source:
            with torch.accelerator.device_index(self.worker.device.index):
                stream = torch.cuda.Stream(device=self.worker.device)
                try:
                    with (
                        torch.cuda.stream(stream),
                        set_current_vllm_config(self.config),
                    ):
                        result = load_local_layers(
                            self.config,
                            tx.new_layers,
                            self.worker.device,
                            self.weight_cache,
                        )
                finally:
                    stream.synchronize()
        tx.weights_ready = True
        return result

    def load_and_transfer(self, log_dir=None):
        return overlap_weight_load_and_kv(
            self.load_weights, self.kv, self.stage, log_dir
        )

    def kv(self):
        tx = self._transaction()
        transferred = 0
        for index in tx.moving:
            name = f"model.layers.{index}.self_attn.attn"
            transferred += tx.transport.copy(
                tx.old_kv.get(name) if self.stage == tx.source else None,
                tx.new_kv.get(name) if self.stage != tx.source else None,
                tx.source,
            )
        torch.cuda.current_stream(self.worker.device).synchronize()
        tx.kv_ready = True
        return {
            "bytes": transferred,
            "block_mapping": "identity",
            "num_blocks": tx.new_config.num_blocks,
        }

    def _bind(self, context, config, caches):
        runner = self.runner
        forward_context = runner.compilation_config.static_forward_context
        forward_context.clear()
        forward_context.update(context)
        runner.kv_cache_config = config
        runner.kv_caches = []
        bind_kv_cache(caches, forward_context, runner.kv_caches)
        if self.worker.use_v2_model_runner:
            from vllm.v1.worker.gpu.attn_utils import init_attn_backend

            runner.attn_groups, _, runner.kernel_block_sizes = init_attn_backend(
                config,
                self.config,
                self.worker.device,
            )
            runner._init_kv_zero_meta()
        else:
            runner.attn_groups = []
            runner.initialize_attn_backend(config)
            runner.initialize_metadata_builders(config, runner._kernel_block_sizes)
            runner._init_kv_zero_meta()

    def commit(self):
        tx = self._transaction()
        if not (tx.weights_ready and tx.kv_ready):
            raise ValueError("weight loading and KV transfer must both complete")
        for index, layer in tx.new_layers.items():
            self.model.layers[index] = layer
        start, end = layer_range(
            self.stage, tx.boundary, self.model.config.num_hidden_layers
        )
        for index in range(self.model.config.num_hidden_layers):
            if not start <= index < end:
                self.model.layers[index] = PPMissingLayer()
        self.model.start_layer, self.model.end_layer = start, end
        self._bind(tx.new_context, tx.new_config, tx.new_kv)
        tx.committed = True
        partition = self._set_partition(tx.boundary)
        return {
            "start_layer": start,
            "end_layer": end,
            "num_blocks": tx.new_config.num_blocks,
            "kv_capacity_bytes": tx.new_config.kv_cache_tensors[0].size,
            "partition": partition,
        }

    def rollback(self):
        tx = self.prepared
        if tx is not None:
            for index, layer in enumerate(tx.old_layers):
                self.model.layers[index] = layer
            self.model.start_layer, self.model.end_layer = layer_range(
                self.stage, tx.old_boundary, self.model.config.num_hidden_layers
            )
            self._bind(tx.old_context, tx.old_config, tx.old_kv)
        if tx is not None:
            tx.transport.close()
        self.prepared = None
        del tx
        gc.collect()
        torch.accelerator.empty_cache()
        boundary = self.model.end_layer if self.stage == 0 else self.model.start_layer
        return {"rolled_back": True, "partition": self._set_partition(boundary)}

    def _set_partition(self, boundary):
        partition = f"{boundary},{self.model.config.num_hidden_layers - boundary}"
        os.environ["VLLM_PP_LAYER_PARTITION"] = partition
        # Workers cache env reads after startup; override this value explicitly.
        envs.VLLM_PP_LAYER_PARTITION = partition
        return partition

    def finalize(self):
        tx = self._transaction()
        if not tx.committed:
            raise ValueError("cannot finalize before commit")
        tx.transport.close()
        self.prepared = None
        del tx
        # Startup freezes weight/loader cycles; retired weights must be collectable.
        gc.unfreeze()
        gc.collect()
        torch.accelerator.empty_cache()
        return {"finalized": True}

    def _transaction(self) -> Prepared:
        if self.prepared is None:
            raise ValueError("no prepared migration")
        return self.prepared


def _layer_index(name):
    return int(name.split(".")[2])


def overlap_weight_load_and_kv(load_weights, transfer_kv, stage, log_dir=None):
    """Join both operations before returning success or permitting rollback."""

    def run(phase, operation):
        started = time.perf_counter_ns()
        if log_dir:
            write_event(log_dir, phase, event="begin", stage=stage, started_ns=started)
        try:
            result = {"ok": True, **operation()}
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finished = time.perf_counter_ns()
        result.update(
            started_ns=started,
            finished_ns=finished,
            elapsed_ms=(finished - started) / 1e6,
        )
        if log_dir:
            write_event(log_dir, phase, event="end", stage=stage, **result)
        return result

    with ThreadPoolExecutor(max_workers=1) as executor:
        weights = executor.submit(run, "weight_load", load_weights)
        kv_result = run("kv", transfer_kv)
        weight_result = weights.result()
    overlap_ms = (
        max(
            0,
            min(weight_result["finished_ns"], kv_result["finished_ns"])
            - max(weight_result["started_ns"], kv_result["started_ns"]),
        )
        / 1e6
    )
    result = {
        "ok": weight_result["ok"] and kv_result["ok"],
        "weight_load": weight_result,
        "kv": kv_result,
        "overlap_ms": overlap_ms,
    }
    if not result["ok"]:
        result["error"] = str(
            [r["error"] for r in (weight_result, kv_result) if not r["ok"]]
        )
    return result


def migration_rpc(worker, action, boundary=None, log_dir=None):
    started = time.perf_counter()
    try:
        if not hasattr(worker, "_molink_migration"):
            worker._molink_migration = LayerMigration(worker)
        migration = worker._molink_migration
        if action in ("describe", "prepare", "prepare_live", "prepare_resize_live"):
            result = getattr(migration, action)(boundary)
        elif action in ("start_live", "start_history"):
            result = migration.start_live(
                log_dir, patching=action == "start_live", **boundary
            )
        elif action == "load_and_transfer":
            if isinstance(boundary, dict):
                result = migration.load_and_transfer(log_dir, **boundary)
            else:
                result = migration.load_and_transfer(log_dir)
        elif action in (
            "connect",
            "commit",
            "rollback",
            "finalize",
            "start_cleanup",
            "cleanup_progress",
            "progress_live",
            "prepared_live",
            "commit_resize_live",
            "rollback_resize_live",
            "sync_live",
            "verify_live",
        ):
            result = getattr(migration, action)()
        else:
            raise ValueError(f"unknown migration action {action}")
        result = {"ok": True, **result}
    except Exception as exc:
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "error_category": "validation"
            if isinstance(exc, ReconfigurationValidationError)
            else "internal",
        }
    result.update(
        stage=get_pp_group().rank_in_group,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    if log_dir:
        write_event(log_dir, action, **result)
    return result
