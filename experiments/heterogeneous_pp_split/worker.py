"""Experiment-only worker: per-rank budgets and allocator peak telemetry."""

import json
import multiprocessing
import os

# CUDA reads this during initialization, before the base worker is imported.
_process = multiprocessing.current_process().name
if _process.startswith("VllmWorker-"):
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
        json.loads(os.environ["SPLIT_MPS_THREADS"])[int(_process.rsplit("-", 1)[1])]
    )

import torch  # noqa: E402

from vllm.v1.worker.gpu_worker import Worker  # noqa: E402


class BudgetWorker(Worker):
    def init_device(self):
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
            json.loads(os.environ["SPLIT_MPS_THREADS"])[self.rank]
        )
        budgets = json.loads(os.environ["SPLIT_BUDGET_GIB"])
        total = torch.cuda.get_device_properties(self.local_rank).total_memory
        self.cache_config.gpu_memory_utilization = budgets[self.rank] * 2**30 / total
        super().init_device()

    def molink_monitor(self):
        result = super().molink_monitor()
        # Dense static KV can have multiple tensors; count every allocation.
        capacity = sum(
            t.size for t in self.model_runner.kv_cache_config.kv_cache_tensors
        )
        result["kv_capacity_bytes"] = capacity
        result["kv_used_bytes"] = (
            capacity * result["active_kv_blocks"] // result["num_blocks"]
        )
        result["visible_sms"] = torch.cuda.get_device_properties(
            self.local_rank
        ).multi_processor_count
        with self._molink_monitor.lock:
            result["timing_records"] = list(self._molink_monitor.samples)
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        return result
