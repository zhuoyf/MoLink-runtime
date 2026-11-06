# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PipeLive worker primitives, retaining MoLink's layer binding and recovery."""

import copy
import hashlib
import os
import threading

import torch

from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.pp_reconfigure.cache import StackedKVCache
from vllm.distributed.pp_reconfigure.communication import MigrationChannel
from vllm.distributed.pp_reconfigure.monitor import write_event
from vllm.distributed.pp_reconfigure.patching import DirtyBitmap, copy_slots
from vllm.distributed.pp_reconfigure.worker import (
    LayerMigration,
    Prepared,
    _layer_index,
)
from vllm.model_executor.models.llama import LlamaDecoderLayer
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.kv_cache_interface import KVCacheGroupSpec


class LiveLayerMigration(LayerMigration):
    def __init__(self, worker):
        super().__init__(worker)
        self.channel = MigrationChannel(get_pp_group(), worker.device)
        self.inference_stream = torch.cuda.Stream(device=worker.device, priority=-1)
        self.inference_stream.wait_stream(torch.cuda.current_stream(worker.device))
        torch.cuda.set_stream(self.inference_stream)
        if self.runner.pp_handler is not None:
            self.runner.pp_handler.main_stream = self.inference_stream
        self.cache_lock = threading.RLock()
        self.active = False
        self.gathered = None
        self.last_write = None
        self.preparing = False
        self.resize_pending: tuple[dict, set[StackedKVCache]] | None = None
        self.error = None
        self.anchor = torch.cuda.Event(enable_timing=True)
        self.anchor.record()
        self.anchor.synchronize()

    def describe(self, boundary):
        result = super().describe(boundary)
        k = int(os.environ.get("HETVLLM_PIPELIVE_STACK", "2"))
        if boundary % k or result["old_boundary"] % k:
            raise ValueError("boundary must align to PipeLive stacking factor")
        if not all(hasattr(cache, "_pipelive") for cache in self.runner.kv_caches):
            raise ValueError("PipeLive requires address-based caches at startup")
        result["pipelive"] = True
        pools = {cache._pipelive[0] for cache in self.runner.kv_caches}
        existing = set(self.runner.kv_cache_config.kv_cache_groups[0].layer_names)
        target = {
            f"model.layers.{i}.self_attn.attn"
            for i in range(result["start_layer"], result["end_layer"])
        }
        exemplar = self.model.layers[self.model.start_layer]
        weight_bytes = sum(
            t.numel() * t.element_size() for t in exemplar.state_dict().values()
        )
        incoming_weights = len(target - existing) * weight_bytes
        allocated = sum(pool.allocated_bytes for pool in pools)
        non_kv = torch.accelerator.memory_allocated(self.worker.device) - allocated
        allocator_free = torch.accelerator.memory_reserved(
            self.worker.device
        ) - torch.accelerator.memory_allocated(self.worker.device)
        free, total = torch.accelerator.get_memory_info(self.worker.device)
        budget = min(
            int(total * self.config.cache_config.gpu_memory_utilization),
            non_kv + allocated + free + allocator_free,
        )
        bytes_per_stack = next(iter(pools)).allocation_bytes
        intermediate = len(existing | target) // k * bytes_per_stack
        final = len(target) // k * bytes_per_stack
        cap = self.config.cache_config.num_gpu_blocks_override
        result["max_intermediate_blocks"] = max(
            0, (budget - non_kv - incoming_weights - (32 << 20)) // intermediate
        )
        result["max_final_blocks"] = max(
            0,
            (
                budget
                - non_kv
                - incoming_weights
                + len(existing - target) * weight_bytes
                - (32 << 20)
            )
            // final,
        )
        if cap is not None:
            result["max_intermediate_blocks"] = min(
                result["max_intermediate_blocks"], cap
            )
            result["max_final_blocks"] = min(result["max_final_blocks"], cap)
        return result

    def prepare(self, boundary):
        return self._prepare_live(boundary)

    def connect(self):
        return {"connected": True}

    def load_weights(self):
        self.weight_result = super().load_weights()
        return self.weight_result

    def load_and_transfer(self, log_dir=None):
        self.start_live(log_dir)
        result = self.sync_live()
        return {
            "ok": True,
            "weight_load": self.weight_result,
            "kv": {"bytes": result["bytes"]},
            "overlap_ms": 0,
            **result,
        }

    def prepare_live(self, boundary):
        description = self.describe(boundary)
        if self.prepared is not None or self.preparing:
            raise ValueError("a migration is already prepared")
        self.error = None
        self.preparing = True
        self.last_write = torch.cuda.Event()
        self.last_write.record()

        def prepare():
            try:
                with (
                    torch.accelerator.device_index(self.worker.device.index),
                    torch.cuda.stream(self.channel.stream),
                ):
                    self._prepare_live(boundary)
                    self.channel.stream.synchronize()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.preparing = False

        self.prepare_thread = threading.Thread(target=prepare, daemon=True)
        self.prepare_thread.start()
        return {**description, "preparing": True}

    def prepared_live(self):
        if self.error:
            raise RuntimeError(self.error)
        return {"prepared": self.prepared is not None and not self.preparing}

    def _prepare_live(self, boundary):
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
        old_boundary = description["old_boundary"]
        moving = list(range(min(boundary, old_boundary), max(boundary, old_boundary)))
        source = 0 if boundary < old_boundary else 1
        new_config = copy.deepcopy(old_config)
        spec = old_config.kv_cache_groups[0].kv_cache_spec
        new_config.kv_cache_groups = [KVCacheGroupSpec(names, spec)]
        # Reuse the native configuration planner for layer metadata.
        from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups

        override = self.config.cache_config.num_gpu_blocks_override
        try:
            self.config.cache_config.num_gpu_blocks_override = old_config.num_blocks
            new_config = get_kv_cache_config_from_groups(
                self.config,
                new_config.kv_cache_groups,
                old_config.num_blocks * spec.page_size_bytes * len(names),
            )
        finally:
            self.config.cache_config.num_gpu_blocks_override = override
        new_layers = {}
        incoming = None
        try:
            if self.stage != source:
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
                existing_ids = set(next(iter(old_kv.values()))._pipelive[0].stacks)
                incoming = StackedKVCache(
                    [f"model.layers.{i}.self_attn.attn" for i in moving],
                    spec,
                    max(existing_ids) + 1,
                    self.worker.device,
                    int(os.environ.get("HETVLLM_PIPELIVE_STACK", "2")),
                )
                incoming.remove_blocks(set(incoming.stacks) - existing_ids)
            new_context = {name: context[name] for name in names}
        finally:
            for name in set(context) - old_context.keys():
                del context[name]
        new_kv = {name: old_kv[name] for name in names if name in old_kv}
        if incoming is not None:
            new_kv.update(incoming.handles)
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
            self.channel,
        )
        self.full_done = False
        self.applied = 0
        self.bytes_sent = 0
        self.error = None
        self.finish = threading.Event()
        self.compute_intervals = []
        self.communication_intervals = []
        self.compute_events = None
        self.step_count = 0
        self.gathered = None
        self.log_dir = None
        self.bitmap = DirtyBitmap(
            max(cache.numel() for cache in old_kv.values()) * spec.block_size
        )
        return description

    def start_live(self, log_dir):
        tx = self._transaction()
        self.anchor.record()
        self.anchor.synchronize()
        self.last_write = torch.cuda.Event()
        self.last_write.record()
        self.log_dir = log_dir
        self.channel.rejections = 0
        self.gpu_profiler = None
        if (
            os.environ.get("HETVLLM_PIPELIVE_PROFILE") == "1"
            and self.stage == tx.source
        ):
            self.gpu_profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            self.gpu_profiler.start()
        self.active = True
        self.weights_thread = threading.Thread(
            target=self._run, args=(self.load_weights,), daemon=True
        )
        self.kv_thread = threading.Thread(
            target=self._run,
            args=(self._send_loop if self.stage == tx.source else self._receive_loop,),
            daemon=True,
        )
        self.weights_thread.start()
        self.kv_thread.start()
        return {"started": True, "source": tx.source}

    def _run(self, operation):
        try:
            with (
                torch.accelerator.device_index(self.worker.device.index),
                torch.inference_mode(),
            ):
                operation()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            if self.log_dir:
                write_event(
                    self.log_dir, "background_error", stage=self.stage, error=self.error
                )

    def begin_compute(self):
        if self.active:
            self.compute_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self.compute_events[0].record()

    def end_compute(self):
        if self.active and self.compute_events is not None:
            self.compute_events[1].record()
            self.compute_intervals.append(self.compute_events)
            self.compute_events = None

    def after_step(self, scheduler_output):
        tx = self._transaction()
        self.step_count += bool(scheduler_output.total_num_scheduled_tokens)
        if self.stage == tx.source and scheduler_output.total_num_scheduled_tokens:
            handle = tx.old_kv[f"model.layers.{tx.moving[0]}.self_attn.attn"]
            pool, _ = handle._pipelive
            slots = pool.last_slots
            slots = slots[slots >= 0].cpu().tolist()
            self.bitmap.mark(slots)
            copied_slots = [
                block.dst_block_id * pool.spec.block_size + token
                for block in (scheduler_output.kv_cache_block_copies or ())
                for token in range(pool.spec.block_size)
            ]
            self.bitmap.mark(copied_slots, advance=False)
            self.last_write = torch.cuda.Event()
            self.last_write.record()

    def _send_snapshot(self, slots, watermark, full=False):
        tx = self._transaction()
        spec = tx.old_config.kv_cache_groups[0].kv_cache_spec
        slot_bytes = (
            spec.num_kv_heads
            * 2
            * spec.head_size
            * torch.empty((), dtype=spec.dtype).element_size()
        )
        chunk_slots = max(1, min(16384, (16 << 20) // slot_bytes))
        chunks = [
            slots[i : i + chunk_slots] for i in range(0, len(slots), chunk_slots)
        ] or [[]]
        for layer_index in tx.moving:
            name = f"model.layers.{layer_index}.self_attn.attn"
            for index, chunk in enumerate(chunks):
                with self.cache_lock, torch.cuda.stream(self.channel.stream):
                    if self.last_write is not None:
                        self.channel.stream.wait_event(self.last_write)
                    payload = copy_slots(tx.old_kv[name], chunk)
                    self.gathered = torch.cuda.Event()
                    self.gathered.record()
                metadata = {
                    "layer": name,
                    "slots": chunk,
                    "watermark": watermark,
                    "last": layer_index == tx.moving[-1] and index == len(chunks) - 1,
                    "full": full,
                    "shape": list(payload.shape),
                    "dtype": str(payload.dtype).removeprefix("torch."),
                }
                events = self.channel.send(payload, metadata)
                self.communication_intervals.append(events)
                self.bytes_sent += payload.numel() * payload.element_size()
                if self.log_dir:
                    write_event(
                        self.log_dir,
                        "patch",
                        stage=self.stage,
                        slots=len(chunk),
                        bytes=payload.numel() * payload.element_size(),
                        watermark=watermark,
                        full=full,
                        last=metadata["last"],
                    )
        self.applied = watermark
        self.full_done |= full

    def _send_loop(self):
        tx = self._transaction()
        pool, _ = tx.old_kv[f"model.layers.{tx.moving[0]}.self_attn.attn"]._pipelive
        block_size = pool.spec.block_size
        slots = [
            block * block_size + token
            for block in sorted(pool.stacks)
            for token in range(block_size)
        ]
        self._send_snapshot(slots, 0, full=True)
        while True:
            with self.cache_lock:
                slots, watermark = self.bitmap.drain()
            if slots or watermark != self.applied:
                self._send_snapshot(slots, watermark)
            elif self.finish.is_set():
                break
            else:
                self.finish.wait(0.001)
        self.channel.send_control({"op": "STOP"})
        if self.channel.receive_control()["op"] != "STOPPED":
            raise RuntimeError("receiver did not acknowledge final synchronization")
        tx.kv_ready = True

    def _receive_loop(self):
        tx = self._transaction()

        def apply(payload, metadata):
            copy_slots(tx.new_kv[metadata["layer"]], metadata["slots"], payload)
            self.channel.stream.synchronize()
            self.bytes_sent += payload.numel() * payload.element_size()
            if metadata["last"]:
                self.applied = metadata["watermark"]
                self.full_done |= metadata["full"]

        while self.channel.receive(apply):
            pass
        tx.kv_ready = True

    def progress_live(self):
        if self.error:
            raise RuntimeError(self.error)
        tx = self._transaction()
        return {
            "full_done": self.full_done,
            "weights_ready": tx.weights_ready,
            "scheduled": self.bitmap.scheduled,
            "applied": self.applied,
            "steps": self.step_count,
            "bytes": self.bytes_sent,
            "source": tx.source,
        }

    def verify_live(self):
        tx = self._transaction()
        if not tx.kv_ready:
            raise ValueError("KV verification requires final synchronization")
        digest = hashlib.sha256()
        caches = tx.old_kv if self.stage == tx.source else tx.new_kv
        for index in tx.moving:
            name = f"model.layers.{index}.self_attn.attn"
            pool, _ = caches[name]._pipelive
            for block_id in sorted(pool.stacks):
                block = pool.block(name, block_id).contiguous().view(torch.uint8)
                digest.update(block.cpu().numpy().tobytes())
        return {"kv_digest": digest.hexdigest()}

    def sync_live(self):
        self.finish.set()
        for thread in (self.kv_thread, self.weights_thread):
            thread.join(timeout=60)
            if thread.is_alive():
                raise TimeoutError("PipeLive background thread did not quiesce")
        self.active = False
        if self.error:
            raise RuntimeError(self.error)
        torch.accelerator.synchronize(self.worker.device)
        gpu_trace = None
        if self.gpu_profiler is not None:
            self.gpu_profiler.stop()
            gpu_trace = f"{self.log_dir}/gpu-stage{self.stage}.json"
            self.gpu_profiler.export_chrome_trace(gpu_trace)
            self.gpu_profiler = None
        result = self.progress_live()
        result.update(
            gpu_trace=gpu_trace,
            compute_intervals_ms=[
                (self.anchor.elapsed_time(a), self.anchor.elapsed_time(b))
                for a, b in self.compute_intervals
            ],
            communication_intervals_ms=[
                (self.anchor.elapsed_time(a), self.anchor.elapsed_time(b))
                for a, b in self.communication_intervals
            ],
            handshake_rejections=self.channel.rejections,
        )
        return result

    def rollback(self):
        if self.preparing:
            self.prepare_thread.join(timeout=60)
            if self.prepare_thread.is_alive():
                raise TimeoutError("PipeLive preparation did not quiesce")
        if self.active:
            self.sync_live()
        return super().rollback()

    def prepare_resize_live(self, plan):
        if self.prepared is not None:
            raise ValueError("cannot resize during KV migration")
        pools = {cache._pipelive[0] for cache in self.runner.kv_caches}
        for pool in pools:
            if (
                not set(plan["remove"]) <= pool.stacks.keys()
                or set(plan["add"]) & pool.stacks.keys()
            ):
                raise ValueError(
                    "worker KV allocation disagrees with scheduler resize plan"
                )
        self.resize_pending = (plan, pools)
        try:
            for pool in pools:
                if plan["add"]:
                    pool.add_blocks(plan["add"])
            torch.cuda.current_stream(self.worker.device).synchronize()
        except Exception:
            self.rollback_resize_live()
            raise
        return {"prepared_resize": True}

    def rollback_resize_live(self):
        if pending := getattr(self, "resize_pending", None):
            plan, pools = pending
            torch.accelerator.synchronize(self.worker.device)
            for pool in pools:
                added = set(plan["add"]) & pool.stacks.keys()
                if added:
                    pool.remove_blocks(added)
            self.resize_pending = None
        return {"resize_rolled_back": True}

    def commit_resize_live(self):
        if self.resize_pending is None:
            raise ValueError("no prepared KV resize")
        plan, pools = self.resize_pending
        torch.accelerator.synchronize(self.worker.device)
        for pool in pools:
            if plan["remove"]:
                pool.remove_blocks(plan["remove"])
        self.runner.kv_cache_config.num_blocks = plan["capacity"]
        self.resize_pending = None
        return {
            "num_blocks": plan["capacity"],
            "allocated_bytes": sum(pool.allocated_bytes for pool in pools),
        }

    def finalize(self):
        tx = self._transaction()
        if not tx.committed:
            raise ValueError("cannot finalize before commit")
        if self.stage == tx.source:
            names = [f"model.layers.{i}.self_attn.attn" for i in tx.moving]
            pools = {tx.old_kv[name]._pipelive[0] for name in names}
            for pool in pools:
                pool.retire_layers(names)
        return super().finalize()
