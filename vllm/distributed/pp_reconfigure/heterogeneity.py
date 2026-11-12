# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in experimental stage delays and physical KV budgets (PP2/eager)."""

import math
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HeterogeneitySimulator:
    decode_ms_per_layer: float = 0
    kv_budget_bytes: int = 2 << 30

    counters: dict = field(
        default_factory=lambda: {
            "prefill_tokens": 0,
            "new_requests": 0,
            "resumed_requests": 0,
            "decode_steps": 0,
            "injected_seconds": 0.0,
        },
        compare=False,
    )

    def __post_init__(self):
        if (
            not math.isfinite(self.decode_ms_per_layer)
            or not 0 <= self.decode_ms_per_layer <= 20
        ):
            raise ValueError("decode delay must be finite and in [0, 20] ms/layer")
        if type(self.kv_budget_bytes) is not int or self.kv_budget_bytes < (64 << 20):
            raise ValueError("KV budget must be an integer >= 64MiB")

    def delay(self, output, layers):
        self.counters["prefill_tokens"] += sum(
            n for n in output.num_scheduled_tokens.values() if n > 1
        )
        self.counters["new_requests"] += len(output.scheduled_new_reqs)
        self.counters["resumed_requests"] += len(
            output.scheduled_cached_reqs.resumed_req_ids
        )
        if output.scheduled_new_reqs or any(
            n != 1 for n in output.num_scheduled_tokens.values()
        ):
            return
        self.counters["decode_steps"] += 1
        if self.decode_ms_per_layer and output.total_num_scheduled_tokens:
            self.counters["injected_seconds"] += (
                self.decode_ms_per_layer * layers / 1000
            )
            # This boundary holds no cache/NCCL lock. Sleep releases the GIL.
            time.sleep(self.decode_ms_per_layer * layers / 1000)

    def capacity(self, layers, bytes_per_layer_block, cap):
        return min(cap, int(self.kv_budget_bytes // (layers * bytes_per_layer_block)))
