# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
import uuid
from pathlib import Path

from vllm.distributed.pp_reconfigure.monitor import write_event


def validate_workers(workers):
    if len(workers) != 2 or {w.get("stage") for w in workers} != {0, 1}:
        raise ValueError("expected exactly two pipeline workers")
    failures = [w for w in workers if not w.get("ok")]
    if failures:
        raise ValueError(str(failures))
    for key in (
        "old_boundary",
        "boundary",
        "num_blocks",
        "spec",
        "layout",
        "transport",
        "weight_manifest",
    ):
        if workers[0].get(key) != workers[1].get(key):
            raise ValueError(f"pipeline workers disagree on {key}")


class ReconfigurationController:
    """Serialize manual transactions using vLLM's existing engine control plane."""

    def __init__(self, engine, log_root):
        self.engine = engine
        self.log_root = Path(log_root)
        self.lock = asyncio.Lock()

    async def state(self, include_workers=False):
        return await self.engine.engine_core.call_utility_async(
            "molink_state", include_workers
        )

    async def reconfigure(self, boundary, mode="sync", verify_kv=False):
        if mode not in ("sync", "async"):
            raise ValueError("migration mode must be sync or async")
        if self.lock.locked():
            raise ValueError("a reconfiguration is already running")
        async with self.lock:
            run_id = uuid.uuid4().hex
            directory = str(self.log_root / run_id)
            timings = {}
            pause_attempted = False
            paused = False
            finalized = False
            background_started = False
            transaction_started = time.perf_counter()
            pause_started = None
            live_stats = None

            async def phase(name, operation):
                start = time.perf_counter()
                write_event(directory, name, event="begin", boundary=boundary)
                try:
                    result = await operation()
                except BaseException as exc:
                    write_event(directory, name, event="error", error=str(exc))
                    raise
                finally:
                    timings[name] = (time.perf_counter() - start) * 1000
                write_event(
                    directory,
                    name,
                    event="end",
                    elapsed_ms=timings[name],
                    result=result,
                )
                return result

            async def rpc(action):
                workers = await self.engine.engine_core.call_utility_async(
                    "molink_migrate", action, boundary, directory
                )
                validate_workers(workers)
                return workers

            try:
                if await self.engine.is_paused():
                    raise ValueError("engine is already paused")
                descriptions = await phase("validate", lambda: rpc("describe"))
                resize_restore = None
                if all(w.get("pipelive") for w in descriptions):
                    shrink = min(w["max_intermediate_blocks"] for w in descriptions)
                    resize_restore = min(w["max_final_blocks"] for w in descriptions)
                    if shrink < descriptions[0]["num_blocks"]:
                        await phase(
                            "resize_before",
                            lambda: self.engine.engine_core.call_utility_async(
                                "molink_resize_kv", shrink
                            ),
                        )
                if mode == "async":
                    if not all(w.get("pipelive") for w in descriptions):
                        raise ValueError(
                            "async migration requires HETVLLM_PIPELIVE=1 at startup"
                        )
                    background_started = True
                    await phase("prepare_live", lambda: rpc("prepare_live"))
                    prepare_deadline = time.monotonic() + 180
                    while not all(w["prepared"] for w in await rpc("prepared_live")):
                        if time.monotonic() >= prepare_deadline:
                            raise TimeoutError(
                                "background KV allocation did not complete"
                            )
                        await asyncio.sleep(0.01)
                    await phase("start_live", lambda: rpc("start_live"))
                    deadline = time.monotonic() + 180
                    while True:
                        progress = await rpc("progress_live")
                        source = next(w for w in progress if w["stage"] == w["source"])
                        destination = next(
                            w for w in progress if w["stage"] != w["source"]
                        )
                        write_event(directory, "progress", workers=progress)
                        if (
                            all(w["full_done"] and w["weights_ready"] for w in progress)
                            and source["scheduled"] - destination["applied"] < 50
                        ):
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError("KV patching did not converge")
                        await asyncio.sleep(0.01)
                pause_attempted = True
                pause_started = time.perf_counter()
                await phase(
                    "pause",
                    lambda: self.engine.pause_generation(
                        mode="keep", clear_cache=False
                    ),
                )
                paused = True
                before = await self.state()
                if before["batch_queue_depth"]:
                    raise RuntimeError("pipeline batches did not drain")
                write_event(directory, "state", event="before", **before)
                if mode == "async":
                    live_stats = await phase("sync_live", lambda: rpc("sync_live"))
                    parallel = live_stats
                else:
                    for action in ("prepare", "connect"):
                        await phase(action, lambda action=action: rpc(action))
                    parallel = await phase(
                        "load_and_transfer", lambda: rpc("load_and_transfer")
                    )
                    if all(w.get("pipelive") for w in descriptions):
                        live_stats = parallel
                verification = None
                if verify_kv:
                    verification = await phase(
                        "verify_live", lambda: rpc("verify_live")
                    )
                    if len({w["kv_digest"] for w in verification}) != 1:
                        raise RuntimeError("source/destination KV digest mismatch")
                await phase("commit", lambda: rpc("commit"))
                after = await self.state()
                for request_id, request in before["requests"].items():
                    if (
                        request_id in after["requests"]
                        and after["requests"][request_id] != request
                    ):
                        raise RuntimeError(f"request/KV mapping changed: {request_id}")
                write_event(directory, "state", event="after", **after)
                # After release begins, rollback is no longer safe on every rank.
                finalized = True
                await phase("finalize", lambda: rpc("finalize"))
                if resize_restore is not None:
                    current = await self.state()
                    if resize_restore != current.get("num_blocks", resize_restore):
                        await phase(
                            "resize_after",
                            lambda: self.engine.engine_core.call_utility_async(
                                "molink_resize_kv", resize_restore
                            ),
                        )
                await phase("resume", self.engine.resume_generation)
                paused = False
                pause_ms = (time.perf_counter() - pause_started) * 1000
                total_ms = (time.perf_counter() - transaction_started) * 1000
                result = {
                    "status": "resumed",
                    "run_id": run_id,
                    "boundary": boundary,
                    "timings_ms": timings,
                    "load_and_transfer": parallel,
                    "log_dir": directory,
                    "snapshot": await self.state(True),
                    "mode": mode,
                    "total_ms": total_ms,
                    "pause_ms": pause_ms,
                    "kv_verification": verification,
                }
                if live_stats is not None:
                    source_stats = next(
                        w for w in live_stats if w["stage"] == w["source"]
                    )
                    communication = source_stats["communication_intervals_ms"]
                    computation = source_stats["compute_intervals_ms"]
                    communication_ms = sum(end - start for start, end in communication)
                    overlap_ms = sum(
                        max(0, min(ce, ge) - max(cs, gs))
                        for cs, ce in communication
                        for gs, ge in computation
                    )
                    result["metrics"] = {
                        "communication_ms": communication_ms,
                        "communication_bytes": source_stats["bytes"],
                        "compute_communication_overlap_ms": overlap_ms,
                        "overlap_fraction": overlap_ms / communication_ms
                        if communication_ms
                        else 0,
                        "inference_steps_during_migration": source_stats["steps"],
                    }
                write_event(directory, "transaction", **result)
                return result
            except BaseException as exc:
                recovery = "unchanged"
                if background_started and not paused:
                    try:
                        pause_attempted = True
                        await self.engine.pause_generation(
                            mode="keep", clear_cache=False
                        )
                        paused = True
                    except Exception as recovery_error:
                        recovery = (
                            f"could not pause background migration: {recovery_error}"
                        )
                if paused and not finalized:
                    try:
                        await phase("rollback", lambda: rpc("rollback"))
                        await phase("resume", self.engine.resume_generation)
                        paused = False
                        recovery = "rolled_back_and_resumed"
                    except Exception as recovery_error:
                        recovery = f"paused: {recovery_error}"
                elif pause_attempted:
                    recovery = "pause state requires inspection"
                write_event(
                    directory,
                    "transaction",
                    status="failed",
                    error=str(exc),
                    recovery=recovery,
                    timings_ms=timings,
                )
                raise RuntimeError(
                    f"{exc}; recovery={recovery}; logs={directory}"
                ) from exc
