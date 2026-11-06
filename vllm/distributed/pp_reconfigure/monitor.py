# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import socket
import threading
import time
from collections import deque
from pathlib import Path

import torch


def write_event(directory: str, phase: str, **fields) -> None:
    """One JSONL log per phase, usable on workers and the head process."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"{phase}.log").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "phase": phase, **fields}) + "\n")


def memory_metrics(allocated, reserved, kv_capacity, kv_used, total):
    # Allocated includes unused KV pages; reserved includes allocator slack too.
    active = max(0, allocated - kv_capacity) + kv_used
    return {
        "torch_allocated_bytes": allocated,
        "torch_reserved_bytes": reserved,
        "kv_capacity_bytes": kv_capacity,
        "kv_used_bytes": kv_used,
        "active_memory_estimate_bytes": active,
        "active_memory_estimate_fraction": active / total if total else None,
    }


class StageMonitor:
    """Nonblocking CUDA event timings; NVML counters are read only on polling."""

    def __init__(self):
        self.pending: deque[tuple[torch.cuda.Event, torch.cuda.Event, float, int]] = (
            deque()
        )
        self.samples: deque[dict[str, float | int]] = deque(maxlen=256)
        self.lock = threading.Lock()
        self.request_blocks: dict[str, set[int]] = {}
        self.kv_state_time = time.time()
        self.steps = 0
        self.dropped = 0

    def begin(self):
        with self.lock:
            self._collect()
            if len(self.pending) >= 256:
                self.dropped += 1
                return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end, time.perf_counter()

    def end(self, events, tokens):
        if events is not None:
            events[1].record()
        with self.lock:
            if events is not None:
                self.pending.append((*events, tokens))
            self.steps += 1

    def collect(self):
        with self.lock:
            self._collect()

    def _collect(self):
        while self.pending and self.pending[0][1].query():
            start, end, host_start, tokens = self.pending.popleft()
            self.samples.append({"gpu_ms": start.elapsed_time(end), "tokens": tokens})

    def update_requests(self, output):
        """Mirror scheduler block assignments on CPU, including shared prefixes."""
        with self.lock:
            for req_id in output.finished_req_ids | (output.preempted_req_ids or set()):
                self.request_blocks.pop(req_id, None)
            for request in output.scheduled_new_reqs:
                self.request_blocks[request.req_id] = set(request.block_ids[0])
            cached = output.scheduled_cached_reqs
            for req_id, blocks in zip(cached.req_ids, cached.new_block_ids):
                if blocks is None:
                    continue
                if req_id in cached.resumed_req_ids:
                    self.request_blocks[req_id] = set(blocks[0])
                else:
                    self.request_blocks.setdefault(req_id, set()).update(blocks[0])
            self.kv_state_time = time.time()

    def snapshot(self, worker):
        with self.lock:
            self._collect()
            samples = list(self.samples)
            steps, dropped = self.steps, self.dropped
            used_blocks = len(set().union(*self.request_blocks.values()) - {0})
            active_requests = len(self.request_blocks)
            kv_state_time = self.kv_state_time
        from vllm.distributed.parallel_state import get_pp_group
        from vllm.utils.import_utils import import_pynvml

        runner = worker.model_runner
        config = runner.kv_cache_config
        capacity = sum(t.size for t in config.kv_cache_tensors[:1])
        if runner.kv_caches and hasattr(runner.kv_caches[0], "_pipelive"):
            pools = {cache._pipelive[0] for cache in runner.kv_caches}
            capacity = sum(pool.allocated_bytes for pool in pools)
        decoder = getattr(runner.get_model(), "model", None)
        properties = torch.cuda.get_device_properties(worker.device)
        total = properties.total_memory
        result = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_uuid": str(properties.uuid),
            "gpu_name": properties.name,
            "device_index": worker.device.index,
            "stage": get_pp_group().rank_in_group,
            "sample_time": time.time(),
            "steps": steps,
            "timing_samples": len(samples),
            "timing_dropped": dropped,
            "stage_gpu_ms_mean": (
                sum(s["gpu_ms"] for s in samples) / len(samples) if samples else None
            ),
            "stage_last": samples[-1] if samples else None,
            "num_blocks": config.num_blocks,
            "layer_range": [
                getattr(decoder, "start_layer", None),
                getattr(decoder, "end_layer", None),
            ],
            "active_requests": active_requests,
            "active_kv_blocks": used_blocks,
            "kv_cache_usage": used_blocks / max(1, config.num_blocks - 1),
            "kv_usage_source": "worker_block_assignments",
            "kv_state_time": kv_state_time,
            "total_memory_bytes": total,
            **memory_metrics(
                torch.accelerator.memory_allocated(worker.device),
                torch.accelerator.memory_reserved(worker.device),
                capacity,
                capacity * used_blocks // config.num_blocks,
                total,
            ),
        }
        try:
            nvml = import_pynvml()
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByUUID(f"GPU-{properties.uuid}")
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            result.update(
                gpu_utilization_percent=util.gpu,
                memory_bandwidth_utilization_percent=util.memory,
                device_memory_used_bytes=nvml.nvmlDeviceGetMemoryInfo(handle).used,
            )
        except Exception as exc:
            result["nvml_error"] = str(exc)
        return result
