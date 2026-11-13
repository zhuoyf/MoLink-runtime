# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import math
import time
import uuid
from dataclasses import asdict

from vllm.distributed.pp_reconfigure.errors import ReconfigurationValidationError
from vllm.distributed.pp_reconfigure.monitor import write_event

from .model import NodeState, PerformanceModel, RuntimeState
from .optimizer import PartitionOptimizer


class AdaptiveScheduler:
    """Head-side task. Worker sampling and all model/log I/O stay off inference."""

    def __init__(self, controller, profile_path, log_dir, config=None, interval=5):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("decision interval must be positive")
        self.controller = controller
        self.profile_path = profile_path
        self.model = PerformanceModel.load(profile_path)
        self.optimizer = PartitionOptimizer(config)
        if self.optimizer.config.security_mode != "off":
            raise ValueError("security mode requires a trusted execution provider")
        self.log_dir = log_dir
        self.interval = interval
        self.previous = None
        self.not_before = 0.0
        self.seen = {}
        self.latest = None
        self.task = None
        self.stopping = False
        self.admission_state = "NORMAL"
        self.admission_lock = asyncio.Lock()

    def ingest(self, reports, version):
        now = time.time()
        workers = sorted(
            (r["metrics"] for r in reports["workers"]), key=lambda w: w["stage"]
        )
        if len(workers) != 2 or any(
            "sample_error" in w
            or not 0 <= now - w["sample_time"] <= self.optimizer.config.stale_seconds
            for w in workers
        ):
            raise ValueError("missing or stale node samples")
        network_seconds = reports.get("publish_seconds")
        if network_seconds and 0 < network_seconds < 1:
            # End-to-end small-message NCCL latency includes host staging.
            self.model.update(
                "network/latency_seconds",
                network_seconds,
                min(w["sample_time"] for w in workers),
            )
        expected_gpus = self.model.conditions.get("gpu_uuids")
        if expected_gpus and expected_gpus != [w["gpu_uuid"] for w in workers]:
            raise ValueError("profile GPU identities do not match this deployment")
        scope = self.model.conditions.get("compute_timing_scope")
        if any(w.get("compute_timing_scope") != scope for w in workers):
            raise ValueError("profile compute timing scope does not match")
        nodes = []
        newest = None
        for w in workers:
            stage = w["stage"]
            layers = w["layer_range"][1] - w["layer_range"][0]
            kv_per_block = w["kv_capacity_bytes"] / layers / w["num_blocks"]
            self.model.update(
                f"kv/{stage}/{layers}/bytes_per_block", kv_per_block, w["sample_time"]
            )
            nodes.append(
                NodeState(
                    stage,
                    layers,
                    w["memory_budget_bytes"],
                    w["torch_allocated_bytes"] - w["kv_capacity_bytes"],
                    w["weight_bytes_per_layer"],
                    kv_per_block,
                    w["num_blocks"],
                    w["active_kv_blocks"],
                    w.get("gpu_utilization_percent", 0),
                    w.get("block_cap"),
                    w.get(
                        "kv_payload_bytes_per_layer_block",
                        self.model.conditions.get("kv_payload_bytes_per_layer_block"),
                    ),
                    kv_budget_bytes=w.get("kv_budget_bytes"),
                )
            )
            for sample in w.get("workload_samples", []):
                if sample["sequence"] <= self.seen.get(stage, -1):
                    continue
                self.seen[stage] = sample["sequence"]
                if sample["timestamp"] < self.not_before:
                    continue
                if (
                    not 0
                    <= now - sample["timestamp"]
                    <= self.optimizer.config.stale_seconds
                ):
                    continue
                if not 0 < sample["gpu_ms"] < 60000:
                    continue
                key = self.model.compute_key(
                    stage,
                    sample["layers"],
                    sample["phase"],
                    sample["batch"],
                    sample["context"],
                    sample["tokens"],
                )
                self.model.update(key, sample["gpu_ms"], sample["timestamp"])
            last = w.get("stage_last")
            if (
                last
                and "batch" in last
                and (newest is None or last["timestamp"] > newest["timestamp"])
            ):
                newest = last
        if (
            newest is None
            or newest["timestamp"] < self.not_before
            or not 0 <= now - newest["timestamp"] <= self.optimizer.config.stale_seconds
        ):
            raise ValueError("no fresh workload sample")
        timestamp = min(w["sample_time"] for w in workers)
        if timestamp < self.not_before:
            raise ValueError("waiting for post-migration node samples")
        expected_layers = self.model.conditions.get("total_layers")
        if expected_layers and sum(n.layers for n in nodes) != expected_layers:
            raise ValueError("inconsistent worker layer snapshots")
        used = max(n.used_blocks for n in nodes)
        growth = 0
        if self.previous and timestamp > self.previous.timestamp:
            growth = (used - max(n.used_blocks for n in self.previous.nodes)) / (
                timestamp - self.previous.timestamp
            )
        self.model.update("load/growth_blocks_per_second", max(0, growth), timestamp)
        growth = self.model.predict("load/growth_blocks_per_second") or 0
        state = RuntimeState(
            timestamp,
            version,
            nodes[0].layers,
            sum(n.layers for n in nodes),
            nodes,
            newest["batch"],
            newest["context"],
            newest["tokens"],
            newest["phase"],
            max(0, growth),
            max(w["preemptions"] for w in workers),
            reports.get("waiting_requests", 0),
            reports["scheduler_paused"],
            newest["tokens"] * self.model.conditions["activation_bytes_per_token"],
            running=reports.get("running_requests"),
        )
        self.previous = state
        return state

    def observe_migration(self, before, target, result):
        source = before.nodes[1 if target > before.boundary else 0]
        kv_bytes = (
            abs(target - before.boundary)
            * source.transfer_bytes_per_block
            * source.used_blocks
        )
        key = self.optimizer.migration_key(before.boundary, target, kv_bytes)
        now = time.time()
        for metric in ("total", "pause"):
            self.model.update(key + "/" + metric, result[metric + "_ms"] / 1000, now)
        metrics = result.get("metrics", {})
        if metrics.get("communication_ms", 0) > 0:
            self.model.update(
                "network/bytes_per_second",
                metrics["communication_bytes"] / (metrics["communication_ms"] / 1000),
                now,
            )
        duration = result["total_ms"] / 1000
        baseline = self.optimizer.evaluate(before, self.model, before.boundary, now)
        active_duration = max(0, duration - result["pause_ms"] / 1000)
        if active_duration > 0 and baseline.get("valid"):
            expected = active_duration / baseline["step_seconds"]
            interference = max(
                0,
                min(
                    1, 1 - metrics.get("inference_steps_during_migration", 0) / expected
                ),
            )
            self.model.update(key + "/interference", interference, now)
        self.optimizer.last_migration = now
        self.not_before = now
        self.previous = None

    def transition(self, state, **details):
        self.admission_state = state
        write_event(self.log_dir, "admission", state=state, **details)

    async def migrate_with_admission(self, state, decision):
        cfg = self.optimizer.config
        rpc = self.controller.engine.engine_core.call_utility_async
        owner = uuid.uuid4().hex
        opened = False
        started = time.monotonic()
        queue_seconds = 0.0
        try:
            async with asyncio.timeout(cfg.admission_timeout), self.controller.lock:
                self.transition("PREPARING", target=decision["target"])
                while True:
                    if self.stopping or self.controller.version != state.version:
                        raise ReconfigurationValidationError(
                            "admission cancelled or configuration changed"
                        )
                    capacity = await rpc(
                        "molink_migration_capacity", decision["target"]
                    )
                    elapsed = time.monotonic() - started
                    if capacity["feasible"]:
                        break
                    if not cfg.admission_control or elapsed >= cfg.admission_timeout:
                        raise ReconfigurationValidationError(
                            "admission capacity timeout"
                        )
                    if not opened:
                        # Cancellation can race the RPC acknowledgement.
                        opened = True
                        await rpc(
                            "molink_admission_gate",
                            "open",
                            owner,
                            cfg.admission_timeout - elapsed,
                        )
                        self.transition("RECLAIMING", capacity=capacity)
                    await asyncio.sleep(cfg.admission_poll_interval)
                    gate = await rpc("molink_admission_gate")
                    if gate["owner"] != owner:
                        raise ReconfigurationValidationError(
                            "admission lease expired or cancelled"
                        )
                    queue_seconds = gate["queue_seconds"]
                    reports = await rpc("molink_resource_state")
                    state = await asyncio.to_thread(self.ingest, reports, state.version)
                    decision = self.optimizer.decide(
                        state,
                        self.model,
                        time.time(),
                        reclaim=True,
                        gate_wait=time.monotonic() - started,
                        queue_seconds=queue_seconds,
                    )
                    self.latest = decision
                    self.transition(
                        "RECLAIMING",
                        decision=decision,
                        queue_seconds=queue_seconds,
                        gate=gate,
                    )
                    if decision["target"] == state.boundary:
                        raise ReconfigurationValidationError(
                            "admission benefit no longer sufficient"
                        )
                if opened:
                    # Bounded independently by EngineCore even if the frontend dies.
                    # Existing executor deadlines are 180 + 180 + 60 seconds.
                    await rpc("molink_admission_gate", "renew", owner, 480)
                self.transition(
                    "MIGRATING",
                    capacity=capacity,
                    gate_wait_seconds=time.monotonic() - started,
                )
            result = await self.controller.reconfigure(
                decision["target"],
                "async",
                expected_version=state.version,
                expected_boundary=state.boundary,
            )
            if opened:
                gate = await rpc("molink_admission_gate")
                queue_seconds = gate.get("queue_seconds", queue_seconds)
            result["admission_queue_seconds"] = queue_seconds
            result["admission_seconds"] = time.monotonic() - started if opened else 0
            return state, decision, result
        finally:
            try:
                self.transition("RECOVERING")
            finally:
                try:
                    if opened:
                        await asyncio.shield(
                            asyncio.wait_for(
                                rpc("molink_admission_gate", "close", owner), timeout=2
                            )
                        )
                finally:
                    self.transition(
                        "NORMAL",
                        duration_seconds=time.monotonic() - started,
                        queue_seconds=queue_seconds,
                    )

    async def tick(self):
        if self.admission_lock.locked():
            return
        async with self.admission_lock:
            await self._tick()

    async def _tick(self):
        if self.controller.lock.locked():
            return
        version = self.controller.version
        reports = await self.controller.engine.engine_core.call_utility_async(
            "molink_resource_state"
        )
        if version != self.controller.version:
            await asyncio.to_thread(
                write_event,
                self.log_dir,
                "decision",
                reason="configuration_changed",
                version=version,
            )
            return
        previous_updates = {k: e.updated for k, e in self.model.estimates.items()}
        state = await asyncio.to_thread(self.ingest, reports, version)
        updates = {
            k: asdict(e)
            for k, e in self.model.estimates.items()
            if e.updated > previous_updates.get(k, -1)
        }
        await asyncio.to_thread(
            write_event,
            self.log_dir,
            "model_update",
            estimates=updates,
            alpha=self.model.alpha,
            rejected=self.model.rejected,
        )
        decision = await asyncio.to_thread(
            self.optimizer.decide, state, self.model, time.time()
        )
        if self.optimizer.config.admission_control and any(
            c.get("reason") in ("intermediate_memory", "target_kv_capacity")
            for c in decision["candidates"]
        ):
            reclaim = self.optimizer.decide(
                state,
                self.model,
                time.time(),
                reclaim=True,
                gate_wait=self.optimizer.config.admission_timeout,
            )

            def selected_cost(choice):
                return next(
                    (
                        c.get("cost_seconds", math.inf)
                        for c in choice["candidates"]
                        if c["boundary"] == choice["target"]
                    ),
                    math.inf,
                )

            if selected_cost(reclaim) < selected_cost(decision):
                decision = reclaim
        self.latest = decision
        await asyncio.to_thread(write_event, self.log_dir, "decision", **decision)
        if decision["target"] != state.boundary:
            try:
                state, decision, result = await self.migrate_with_admission(
                    state, decision
                )
            except (ReconfigurationValidationError, TimeoutError) as exc:
                await asyncio.to_thread(
                    write_event,
                    self.log_dir,
                    "outcome",
                    status="validation_rejected",
                    error=str(exc),
                    decision=decision,
                )
            except Exception as exc:
                self.optimizer.failed = True
                await asyncio.to_thread(
                    write_event,
                    self.log_dir,
                    "outcome",
                    status="failed",
                    error=str(exc),
                    decision=decision,
                )
            else:
                await asyncio.to_thread(
                    self.observe_migration, state, decision["target"], result
                )
                await asyncio.to_thread(
                    write_event,
                    self.log_dir,
                    "outcome",
                    status="resumed",
                    result=result,
                    prediction=decision,
                )
        await asyncio.to_thread(self.model.save, self.profile_path)

    async def run(self):
        while not self.stopping:
            try:
                await self.tick()
            except Exception as exc:
                await asyncio.to_thread(
                    write_event, self.log_dir, "sample_error", error=str(exc)
                )
            await asyncio.sleep(self.interval)

    def start(self):
        self.task = asyncio.create_task(self.run())

    async def stop(self):
        self.stopping = True
        if self.task:
            # Do not cancel an in-flight PipeLive transaction.
            await self.task

    def status(self):
        return {
            "running": self.task is not None and not self.task.done(),
            "failure_latched": self.optimizer.failed,
            "admission_state": self.admission_state,
            "latest": self.latest,
            "config": asdict(self.optimizer.config),
            "model_buckets": len(self.model.estimates),
        }
