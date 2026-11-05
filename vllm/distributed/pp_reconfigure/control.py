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

    async def reconfigure(self, boundary):
        if self.lock.locked():
            raise ValueError("a reconfiguration is already running")
        async with self.lock:
            run_id = uuid.uuid4().hex
            directory = str(self.log_root / run_id)
            timings = {}
            pause_attempted = False
            paused = False
            finalized = False

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
                await phase("validate", lambda: rpc("describe"))
                pause_attempted = True
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
                for action in ("prepare", "connect"):
                    await phase(action, lambda action=action: rpc(action))
                parallel = await phase(
                    "load_and_transfer", lambda: rpc("load_and_transfer")
                )
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
                await phase("resume", self.engine.resume_generation)
                paused = False
                result = {
                    "status": "resumed",
                    "run_id": run_id,
                    "boundary": boundary,
                    "timings_ms": timings,
                    "load_and_transfer": parallel,
                    "log_dir": directory,
                    "snapshot": await self.state(True),
                }
                write_event(directory, "transaction", **result)
                return result
            except BaseException as exc:
                recovery = "unchanged"
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
