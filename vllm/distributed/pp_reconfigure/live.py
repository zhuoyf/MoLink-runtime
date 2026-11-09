# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker primitives for online layer migration and recovery."""

import gc
import hashlib
import threading
import time

import torch

import vllm.envs as envs
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.pp_reconfigure.cache import StackedKVCache
from vllm.distributed.pp_reconfigure.communication import MigrationChannel
from vllm.distributed.pp_reconfigure.errors import ReconfigurationValidationError
from vllm.distributed.pp_reconfigure.monitor import write_event
from vllm.distributed.pp_reconfigure.patching import (
    DirtyBitmap,
    PatchLifecycle,
    copy_slots,
)
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
        self.preparing = False
        self.resize_pending: tuple[dict, set[StackedKVCache]] | None = None
        self.error = None
        self.retired = None
        self.cleanup_thread = None
        self.cleanup_error = None
        self.anchor = torch.cuda.Event(enable_timing=True)
        self.anchor.record()
        self.anchor.synchronize()

    def describe(self, boundary):
        self._wait_cleanup()
        result = super().describe(boundary)
        k = envs.VLLM_PP_RECONFIGURE_STACK_SIZE
        if boundary % k or result["old_boundary"] % k:
            raise ReconfigurationValidationError(
                "boundary must align to the layer stacking factor"
            )
        if not all(
            hasattr(cache, "_pp_reconfigure") for cache in self.runner.kv_caches
        ):
            raise ReconfigurationValidationError(
                "Online migration requires address-based caches at startup"
            )
        result["online_reconfiguration"] = True
        pools = {cache._pp_reconfigure[0] for cache in self.runner.kv_caches}
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
        result["capacity_memory"] = {
            "budget_bytes": budget,
            "non_kv_bytes": non_kv,
            "kv_bytes": allocated,
            "incoming_weights_bytes": incoming_weights,
            "device_free_bytes": free,
            "allocator_free_bytes": allocator_free,
            "intermediate_bytes_per_block": intermediate,
            "final_bytes_per_block": final,
        }
        return result

    def connect(self):
        return {"connected": True}

    def load_weights(self):
        if self.log_dir:
            write_event(self.log_dir, "weight_load", event="begin", stage=self.stage)
        self.weight_result = super().load_weights()
        if self.log_dir:
            write_event(
                self.log_dir,
                "weight_load",
                event="end",
                stage=self.stage,
                **self.weight_result,
            )
        return self.weight_result

    def load_and_transfer(self, log_dir=None, block_ids=None, generations=None):
        self.start_live(log_dir, block_ids=block_ids, generations=generations)
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
        old_boundary = description["old_boundary"]
        source = 0 if boundary < old_boundary else 1
        if self.stage != source:
            self.weight_cache.prefetch(
                self.config,
                range(min(boundary, old_boundary), max(boundary, old_boundary)),
            )
        self.error = None
        self.preparing = True

        def prepare():
            try:
                with (
                    torch.accelerator.device_index(self.worker.device.index),
                    torch.cuda.stream(self.channel.stream),
                ):
                    self.prepare(boundary)
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
        old_boundary = description["old_boundary"]
        moving = list(range(min(boundary, old_boundary), max(boundary, old_boundary)))
        source = 0 if boundary < old_boundary else 1
        spec = old_config.kv_cache_groups[0].kv_cache_spec
        # Reuse the native configuration planner for layer metadata.
        from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups

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
                existing_ids = next(iter(old_kv.values()))._pp_reconfigure[0].stacks
                incoming = StackedKVCache(
                    [f"model.layers.{i}.self_attn.attn" for i in moving],
                    spec,
                    0,
                    self.worker.device,
                    envs.VLLM_PP_RECONFIGURE_STACK_SIZE,
                )
                incoming.add_blocks(sorted(existing_ids))
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
        self.transferred_slots = set()
        self.error = None
        self.finish = threading.Event()
        self.compute_intervals = []
        self.communication_intervals = []
        self.patch_intervals = []
        self.patch_rounds = 0
        self.compute_events = None
        self.step_count = 0
        self.gathered = None
        self.log_dir = None
        self.bitmap = DirtyBitmap(
            max(cache.numel() for cache in old_kv.values()) * spec.block_size
        )
        return description

    def start_live(self, log_dir, block_ids=None, generations=None, patching=True):
        tx = self._transaction()
        pool, _ = next(iter(tx.old_kv.values()))._pp_reconfigure
        self.history_block_ids = sorted(pool.stacks if block_ids is None else block_ids)
        if not set(self.history_block_ids) <= pool.stacks.keys():
            raise ValueError("history block manifest references unallocated KV")
        self.lifecycle = PatchLifecycle(
            generations
            if generations is not None
            else [0] * (len(self.bitmap.bits) // pool.spec.block_size)
        )
        self.send_version = 0
        self.patching = patching
        self.stale_slots = 0
        self.anchor.record()
        self.anchor.synchronize()
        self.last_write = torch.cuda.Event()
        self.last_write.record()
        self.log_dir = log_dir
        self.channel.rejections = 0
        self.channel.timings_ms.clear()
        self.gpu_profiler = None
        if envs.VLLM_PP_RECONFIGURE_PROFILE:
            activities = [torch.profiler.ProfilerActivity.CUDA]
            if envs.VLLM_PP_RECONFIGURE_PROFILE_CPU:
                activities.append(torch.profiler.ProfilerActivity.CPU)
            self.gpu_profiler = torch.profiler.profile(activities=activities)
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
        return {
            "started": True,
            "source": tx.source,
            "history_block_ids": self.history_block_ids,
            "allocated_blocks": len(pool.stacks),
        }

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

    def before_step(self, scheduler_output):
        generations = scheduler_output.pp_block_generations
        if generations is None:
            raise ValueError("missing scheduler block generations during migration")
        changed = self.lifecycle.observe(generations)
        tx = self._transaction()
        if changed and self.log_dir:
            write_event(
                self.log_dir,
                "generation",
                stage=self.stage,
                blocks=changed,
                generations=[generations[b] for b in changed],
            )
        if changed and self.stage == tx.source:
            block_size = tx.old_config.kv_cache_groups[0].kv_cache_spec.block_size
            # Refresh the whole reused block, including untouched tail slots.
            self.bitmap.mark(
                [b * block_size + t for b in changed for t in range(block_size)],
                advance=False,
            )

    def after_step(self, scheduler_output):
        tx = self._transaction()
        self.step_count += bool(scheduler_output.total_num_scheduled_tokens)
        if self.stage == tx.source and scheduler_output.total_num_scheduled_tokens:
            handle = tx.old_kv[f"model.layers.{tx.moving[0]}.self_attn.attn"]
            pool, _ = handle._pp_reconfigure
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
            if self.log_dir:
                write_event(
                    self.log_dir,
                    "dirty",
                    stage=self.stage,
                    slots=len(slots),
                    copied_slots=len(copied_slots),
                    scheduled=self.bitmap.scheduled,
                )

    def _send_snapshot(self, slots, watermark, full=False):
        tx = self._transaction()
        spec = tx.old_config.kv_cache_groups[0].kv_cache_spec
        slot_bytes = (
            spec.num_kv_heads
            * 2
            * spec.head_size
            * torch.empty((), dtype=spec.dtype).element_size()
        )
        names = [f"model.layers.{i}.self_attn.attn" for i in tx.moving]
        chunk_slots = max(1, min(16384, (16 << 20) // (slot_bytes * len(names))))
        chunks = [
            slots[i : i + chunk_slots] for i in range(0, len(slots), chunk_slots)
        ] or [[]]
        for index, chunk in enumerate(chunks):
            gather_started = time.perf_counter()
            with self.cache_lock, torch.cuda.stream(self.channel.stream):
                acquired = time.perf_counter()
                self.channel.timings_ms["gather_lock_and_stream_entry"] += (
                    acquired - gather_started
                ) * 1000
                self.channel.stream.wait_event(self.last_write)
                # Snapshot all moving layers before allowing another KV writer.
                payload = torch.stack(
                    [copy_slots(tx.old_kv[name], chunk) for name in names]
                )
                self.gathered = torch.cuda.Event()
                self.gathered.record()
                generations = [
                    (block, self.lifecycle.generations[block])
                    for block in sorted({slot // spec.block_size for slot in chunk})
                ]
                self.channel.timings_ms["gather_enqueue"] += (
                    time.perf_counter() - acquired
                ) * 1000
            self.send_version += 1
            metadata = {
                "transaction": self.log_dir,
                "version": self.send_version,
                "generations": generations,
                "layers": names,
                "slots": chunk,
                "watermark": watermark,
                "last": index == len(chunks) - 1,
                "full": full,
                "shape": list(payload.shape),
                "dtype": str(payload.dtype).removeprefix("torch."),
            }
            if self.log_dir:
                write_event(
                    self.log_dir,
                    "patch_submit",
                    stage=self.stage,
                    slots=len(chunk),
                    watermark=watermark,
                    full=full,
                    last=metadata["last"],
                    layers=names,
                    version=self.send_version,
                    generations=generations,
                )
            events = self.channel.send(payload, metadata)
            self.communication_intervals.append(events)
            if not full and chunk:
                self.patch_intervals.append(events)
            self.bytes_sent += payload.numel() * payload.element_size()
            if self.log_dir:
                write_event(
                    self.log_dir,
                    "patch",
                    event="nccl_complete_and_applied",
                    stage=self.stage,
                    slots=len(chunk),
                    bytes=payload.numel() * payload.element_size(),
                    watermark=watermark,
                    full=full,
                    last=metadata["last"],
                    layers=names,
                )
        self.applied = watermark
        self.transferred_slots.update(slots)
        self.full_done |= full
        self.patch_rounds += bool(slots) and not full

    def _send_loop(self):
        tx = self._transaction()
        pool, _ = tx.old_kv[
            f"model.layers.{tx.moving[0]}.self_attn.attn"
        ]._pp_reconfigure
        block_size = pool.spec.block_size
        slots = [
            block * block_size + token
            for block in self.history_block_ids
            for token in range(block_size)
        ]
        self._send_snapshot(slots, 0, full=True)
        if not self.patching:
            self.finish.wait()
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
            if metadata["transaction"] != self.log_dir:
                raise ValueError("KV patch belongs to another transaction")
            block_size = tx.old_config.kv_cache_groups[0].kv_cache_spec.block_size
            with self.lifecycle.lock:
                valid = self.lifecycle.accept(
                    metadata["version"], metadata["generations"]
                )
                indices = [
                    i
                    for i, slot in enumerate(metadata["slots"])
                    if slot // block_size in valid
                ]
                slots = [metadata["slots"][i] for i in indices]
                if len(indices) != len(metadata["slots"]):
                    self.stale_slots += len(metadata["slots"]) - len(indices)
                    selection = torch.tensor(
                        indices, device=payload.device, dtype=torch.int64
                    )
                    payload_to_apply = payload.index_select(1, selection)
                else:
                    payload_to_apply = payload
                for name, layer_payload in zip(
                    metadata["layers"], payload_to_apply, strict=True
                ):
                    copy_slots(tx.new_kv[name], slots, layer_payload)
                self.channel.stream.synchronize()
            self.transferred_slots.update(metadata["slots"])
            self.bytes_sent += payload.numel() * payload.element_size()
            if self.log_dir:
                write_event(
                    self.log_dir,
                    "patch_applied",
                    stage=self.stage,
                    slots=len(metadata["slots"]),
                    watermark=metadata["watermark"],
                    full=metadata["full"],
                    last=metadata["last"],
                    layers=metadata["layers"],
                    version=metadata["version"],
                    stale_slots=self.stale_slots,
                )
            if metadata["last"]:
                self.patch_rounds += bool(metadata["slots"]) and not metadata["full"]
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
            "dirty_slots": int(self.bitmap.bits.sum()),
            "patch_rounds": self.patch_rounds,
            "kv_ready": tx.kv_ready,
            "steps": self.step_count,
            "bytes": self.bytes_sent,
            "source": tx.source,
            "patch_version": (
                self.send_version if self.stage == tx.source else self.lifecycle.version
            ),
            "stale_slots": self.stale_slots,
        }

    def verify_live(self):
        tx = self._transaction()
        if not tx.kv_ready:
            raise ValueError("KV verification requires final synchronization")
        digest = hashlib.sha256()
        slots = sorted(self.transferred_slots)
        digest.update(str(slots).encode())
        caches = tx.old_kv if self.stage == tx.source else tx.new_kv
        for index in tx.moving:
            name = f"model.layers.{index}.self_attn.attn"
            for start in range(0, len(slots), 1024):
                payload = copy_slots(caches[name], slots[start : start + 1024])
                digest.update(payload.view(torch.uint8).cpu().numpy().tobytes())
        return {"kv_digest": digest.hexdigest(), "verified_slots": len(slots)}

    def _quiesce_live(self):
        self.finish.set()
        for thread in (self.kv_thread, self.weights_thread):
            thread.join(timeout=60)
            if thread.is_alive():
                raise TimeoutError("Online migration background thread did not quiesce")
        self.active = False
        torch.accelerator.synchronize(self.worker.device)
        gpu_trace = None
        if self.gpu_profiler is not None:
            self.gpu_profiler.stop()
            gpu_trace = f"{self.log_dir}/gpu-stage{self.stage}.json"
            self.gpu_profiler.export_chrome_trace(gpu_trace)
            self.gpu_profiler = None
        return gpu_trace

    def sync_live(self):
        gpu_trace = self._quiesce_live()
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
            patch_intervals_ms=[
                (self.anchor.elapsed_time(a), self.anchor.elapsed_time(b))
                for a, b in self.patch_intervals
            ],
            handshake_rejections=self.channel.rejections,
            communication_host_ms=dict(self.channel.timings_ms),
        )
        return result

    def rollback(self):
        if self.preparing:
            self.prepare_thread.join(timeout=60)
            if self.prepare_thread.is_alive():
                raise TimeoutError("Online migration preparation did not quiesce")
        if self.active:
            self._quiesce_live()
        return super().rollback()

    def prepare_resize_live(self, plan):
        self._wait_cleanup()
        if self.prepared is not None:
            raise ValueError("cannot resize during KV migration")
        pools = {cache._pp_reconfigure[0] for cache in self.runner.kv_caches}
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
        if self.resize_pending is not None:
            plan, pools = self.resize_pending
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
        self.retired = tx
        self.prepared = None
        if not envs.VLLM_PP_RECONFIGURE_DEFER_CLEANUP:
            del tx
            self._cleanup_retired()
            if self.cleanup_error:
                raise RuntimeError(self.cleanup_error)
            return {"finalized": True, "cleanup_pending": False}
        return {"finalized": True, "cleanup_pending": True}

    def start_cleanup(self):
        if self.cleanup_thread is not None and self.cleanup_thread.is_alive():
            raise ValueError("previous cleanup is still running")
        self.cleanup_error = None
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_retired, daemon=True
        )
        self.cleanup_thread.start()
        return {"cleanup_started": True}

    def _cleanup_retired(self):
        try:
            with (
                torch.accelerator.device_index(self.worker.device.index),
                torch.inference_mode(),
            ):
                started = time.perf_counter()
                if self.log_dir:
                    write_event(
                        self.log_dir, "cleanup_worker", event="begin", stage=self.stage
                    )
                tx = self.retired
                if tx is not None:
                    if self.stage == tx.source:
                        names = [f"model.layers.{i}.self_attn.attn" for i in tx.moving]
                        pools = {tx.old_kv[name]._pp_reconfigure[0] for name in names}
                        for pool in pools:
                            pool.retire_layers(names)
                    tx.transport.close()
                    self.retired = None
                    del tx
                    gc.unfreeze()
                    gc.collect()
                    torch.accelerator.empty_cache()
                if self.log_dir:
                    write_event(
                        self.log_dir,
                        "cleanup_worker",
                        event="end",
                        stage=self.stage,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                    )
        except Exception as exc:
            self.cleanup_error = f"{type(exc).__name__}: {exc}"

    def cleanup_progress(self):
        if self.cleanup_error:
            raise RuntimeError(self.cleanup_error)
        return {
            "cleanup_done": self.retired is None
            and (self.cleanup_thread is None or not self.cleanup_thread.is_alive())
        }

    def _wait_cleanup(self):
        if self.cleanup_thread is not None:
            self.cleanup_thread.join(timeout=60)
            if self.cleanup_thread.is_alive():
                raise TimeoutError("old resource cleanup did not quiesce")
        if self.cleanup_error:
            raise RuntimeError(self.cleanup_error)
        if self.retired is not None:
            raise ValueError("old resource cleanup has not started")
