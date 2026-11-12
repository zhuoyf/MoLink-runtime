# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import asdict, dataclass

from .model import bucket
from .security import PlacementPlan, SecurityCostModel, SecurityPolicy


@dataclass
class OptimizerConfig:
    stack_size: int = 4
    horizon: float = 60
    cooldown: float = 30
    min_gain: float = 0.05
    stale_seconds: float = 10
    memory_reserve: int = 128 << 20
    security_mode: str = "off"

    def __post_init__(self):
        if self.stack_size < 1 or self.memory_reserve < (128 << 20):
            raise ValueError("invalid stack size or memory reserve")
        if not all(
            math.isfinite(x) and x > 0 for x in (self.horizon, self.stale_seconds)
        ):
            raise ValueError("invalid horizon or staleness limit")
        if (
            not math.isfinite(self.cooldown)
            or self.cooldown < 0
            or not 0 <= self.min_gain < 1
        ):
            raise ValueError("invalid cooldown or gain threshold")


class PartitionOptimizer:
    """Compare lost service seconds over a shared horizon, including migration."""

    def __init__(self, config=None):
        self.config = config or OptimizerConfig()
        self.last_migration = -math.inf
        self.failed = False
        self.security = SecurityPolicy(self.config.security_mode)
        self.security_cost = SecurityCostModel()

    @staticmethod
    def migration_key(old, target, kv_bytes):
        direction = "1to0" if target > old else "0to1"
        return f"migration/{direction}/{abs(target - old)}/{bucket(kv_bytes)}"

    def evaluate(self, state, model, boundary, now):
        cfg = self.config
        plan = PlacementPlan(
            boundary,
            ("tee", "tee") if cfg.security_mode == "trusted" else ("ree", "ree"),
            cfg.security_mode,
        )
        self.security.validate(plan, state.capabilities, now)
        security_seconds = self.security_cost.cost_seconds(plan, state)
        layers = (boundary, state.total_layers - boundary)
        stages = [
            model.compute(
                n.stage,
                layers[n.stage],
                state.phase,
                state.batch,
                state.context,
                state.tokens,
            )
            for n in state.nodes
        ]
        if any(x is None or x <= 0 for x in stages):
            return {
                "boundary": boundary,
                "valid": False,
                "reason": "missing_compute_bucket",
            }
        bandwidth = model.predict("network/bytes_per_second")
        latency = model.predict("network/latency_seconds")
        if not bandwidth or latency is None:
            return {
                "boundary": boundary,
                "valid": False,
                "reason": "missing_network_profile",
            }
        capacities, peaks = [], []
        moving_bytes = 0
        for node in state.nodes:
            target = layers[node.stage]
            incoming = max(0, target - node.layers)
            final_non_kv = (
                node.non_kv_bytes + (target - node.layers) * node.weight_bytes_per_layer
            )
            per_block = target * node.kv_bytes_per_layer_block
            capacity = max(
                0,
                int(
                    (node.memory_budget - final_non_kv - cfg.memory_reserve) / per_block
                ),
            )
            if node.kv_budget_bytes is not None:
                capacity = min(capacity, int(node.kv_budget_bytes / per_block))
            if node.block_cap is not None:
                capacity = min(capacity, node.block_cap)
            if boundary == state.boundary:
                capacity = min(capacity, node.num_blocks)
            capacities.append(capacity)
            # Address-based KV pools coexist only for the union of owned layers.
            peak = (
                node.non_kv_bytes
                + incoming * node.weight_bytes_per_layer
                + max(target, node.layers)
                * node.kv_bytes_per_layer_block
                * node.num_blocks
                + cfg.memory_reserve
            )
            peaks.append(peak)
            if incoming:
                moving_bytes = (
                    incoming * node.transfer_bytes_per_block * node.used_blocks
                )
            if boundary != state.boundary and peak > node.memory_budget:
                return {
                    "boundary": boundary,
                    "valid": False,
                    "reason": "intermediate_memory",
                    "peak_bytes": peaks,
                }
        capacity = min(capacities)
        used = max(n.used_blocks for n in state.nodes)
        projected = (
            used
            + max(0, state.growth_blocks_per_second) * cfg.horizon
            + state.waiting
            * math.ceil(state.context / model.conditions.get("block_size", 16))
        )
        # Risk is a proxy for lost service, separate from measured preemptions.
        risk = min(1.0, max(0, projected - capacity + 1) / max(1, projected))
        if boundary != state.boundary and capacity <= used:
            return {
                "boundary": boundary,
                "valid": False,
                "reason": "target_kv_capacity",
            }
        concurrent = (state.running if state.running is not None else state.batch) > 1
        compute_ms = (
            max(stages) if state.phase == "decode" and concurrent else sum(stages)
        )
        step_seconds = compute_ms / 1000 + latency + state.activation_bytes / bandwidth
        total = pause = interference = 0.0
        if boundary != state.boundary:
            key = self.migration_key(state.boundary, boundary, moving_bytes)
            values = [
                model.migration(key, metric)
                for metric in ("total", "pause", "interference")
            ]
            if any(v is None for v in values):
                return {
                    "boundary": boundary,
                    "valid": False,
                    "reason": "missing_migration_profile",
                    "key": key,
                }
            total, pause, interference = values
            total = max(total, pause)
        phase_predictions = {}
        for phase in ("prefill", "decode"):
            phase_times = [
                model.compute(
                    n.stage,
                    layers[n.stage],
                    phase,
                    state.batch,
                    state.context,
                    state.tokens if phase == "prefill" else state.batch,
                )
                for n in state.nodes
            ]
            phase_predictions[phase] = (
                (
                    max(phase_times)
                    if phase == "decode" and concurrent
                    else sum(phase_times)
                )
                / 1000
                if all(t is not None for t in phase_times)
                else None
            )
        return {
            "phase_predictions_seconds": phase_predictions,
            "boundary": boundary,
            "valid": True,
            "step_seconds": step_seconds,
            "capacity_blocks": capacity,
            "projected_blocks": projected,
            "risk": risk,
            "peak_bytes": peaks,
            "migration_total": total,
            "migration_pause": pause,
            "interference": interference,
            "security_seconds": security_seconds,
        }

    def decide(self, state, model, now):
        result = {
            "state": asdict(state),
            "candidates": [],
            "target": state.boundary,
            "version": state.version,
            "reason": "hold",
        }
        if (
            self.failed
            or state.paused
            or not 0 <= now - state.timestamp <= self.config.stale_seconds
        ):
            result["reason"] = "failure_latched" if self.failed else "paused_or_stale"
            return result
        if len(state.nodes) != 2 or {n.stage for n in state.nodes} != {0, 1}:
            result["reason"] = "missing_nodes"
            return result
        if not state.waiting and (
            state.running == 0
            or (state.running is None and all(n.used_blocks == 0 for n in state.nodes))
        ):
            result["reason"] = "idle"
            return result
        if now - self.last_migration < self.config.cooldown:
            result["reason"] = "cooldown"
            return result
        for boundary in range(
            self.config.stack_size, state.total_layers, self.config.stack_size
        ):
            try:
                candidate = self.evaluate(state, model, boundary, now)
            except ValueError as exc:
                candidate = {"boundary": boundary, "valid": False, "reason": str(exc)}
            result["candidates"].append(candidate)
        current = next(
            (
                c
                for c in result["candidates"]
                if c["boundary"] == state.boundary and c["valid"]
            ),
            None,
        )
        if current is None:
            result["reason"] = "current_prediction_unavailable"
            return result
        for c in result["candidates"]:
            if c["valid"]:
                # The old placement serves until commit; gains start afterwards.
                transition = min(self.config.horizon, c["migration_total"])
                steady = self.config.horizon - transition
                pause = min(transition, c["migration_pause"])
                c["cost_seconds"] = (
                    transition * (1 + current["risk"])
                    + steady * (c["step_seconds"] / current["step_seconds"] + c["risk"])
                    + pause
                    + max(0, transition - pause) * c["interference"]
                    + c["security_seconds"]
                )
        errors = [
            e.relative_error
            for k, e in model.estimates.items()
            if k.startswith("compute/")
            and k.split("/")[3:]
            == model.compute_key(
                0, 1, state.phase, state.batch, state.context, state.tokens
            ).split("/")[3:]
        ]
        result["model_relative_error"] = max(errors, default=0)
        best = min(
            (c for c in result["candidates"] if c["valid"]),
            key=lambda c: c["cost_seconds"],
        )
        gain = (current["cost_seconds"] - best["cost_seconds"]) / current[
            "cost_seconds"
        ]
        result["gain"] = gain
        if best["boundary"] != state.boundary and gain >= self.config.min_gain + min(
            0.5, result["model_relative_error"] * 0.1
        ):
            result.update(target=best["boundary"], reason="expected_service_gain")
        else:
            result["reason"] = "insufficient_gain"
        return result
