# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed-loop HTTP/GPU regression; preserve every request and monitor sample."""

import argparse
import asyncio
import json
import shutil
import statistics
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive_pp"))
from experiment import HTTPController, append, generate  # noqa: E402

from vllm.distributed.pp_reconfigure.adaptive.optimizer import OptimizerConfig
from vllm.distributed.pp_reconfigure.adaptive.runtime import AdaptiveScheduler


async def wait_running(control):
    async with asyncio.timeout(30):
        while True:
            state = await control.state()
            if state["running_requests"]:
                return state
            await asyncio.sleep(0.05)


async def trial(args, concurrency, enabled, repeat):
    directory = args.out / f"{args.case}-c{concurrency}-gate{int(enabled)}-r{repeat}"
    directory.mkdir(parents=True, exist_ok=False)
    append(
        directory / "parameters.jsonl",
        {
            **vars(args),
            "out": str(args.out),
            "concurrency": concurrency,
            "gate": enabled,
            "repeat": repeat,
            "time": time.time(),
        },
    )
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=180)
    ) as client:
        control = HTTPController(client, args.url)
        initial = await control.state()
        boundary = initial["workers"][0]["layer_range"][1]
        if boundary != args.initial_boundary:
            await control.reconfigure(args.initial_boundary)
        if args.case in ("auto", "comparison"):
            await control.request(
                "POST",
                "/molink/heterogeneity",
                {
                    "stages": [
                        {"decode_ms_per_layer": args.delay, "kv_budget_bytes": 2 << 30},
                        {"decode_ms_per_layer": 0, "kv_budget_bytes": 2 << 30},
                    ],
                },
            )
        await control.request("POST", "/molink/resize_kv", {"num_blocks": args.blocks})
        initial = await control.state()
        append(directory / "initial.jsonl", initial)
        baseline = (
            await generate(
                control, 1, args.context, args.tokens, directory / "baseline.jsonl"
            )
        )[0]["token_ids"]
        stop = asyncio.Event()
        records = []
        samples = []
        errors = []
        start = time.time()

        async def slot():
            while not stop.is_set():
                try:
                    records.extend(
                        await generate(
                            control,
                            1,
                            args.context,
                            args.tokens,
                            directory / "requests.jsonl",
                        )
                    )
                except Exception as exc:
                    errors.append(str(exc))
                    append(
                        directory / "errors.jsonl",
                        {"time": time.time(), "error": str(exc)},
                    )
                    stop.set()
                    raise

        async def monitor():
            while not stop.is_set():
                sample = await control.state()
                sample["time"] = time.time()
                samples.append(sample)
                append(directory / "samples.jsonl", sample)
                await asyncio.sleep(0.1)

        slots = [asyncio.create_task(slot()) for _ in range(concurrency)]
        monitoring = asyncio.create_task(monitor())
        owner = uuid.uuid4().hex
        runtime = None
        proof = {}
        try:
            before = await wait_running(control)
            if args.case in ("gate", "memory", "failure"):
                timeout = 0.5 if args.case == "failure" else 10
                await control.call_utility_async(
                    "molink_admission_gate", "open", owner, timeout
                )
                opened = await control.state()
                ids = set(opened["requests"])
                await asyncio.sleep(0.4)
                after = await control.state()
                progress = [
                    rid
                    for rid in ids & after["requests"].keys()
                    if len(after["requests"][rid]["output_token_ids"])
                    > len(opened["requests"][rid]["output_token_ids"])
                ]
                assert all(
                    r["num_computed_tokens"] == 0
                    for rid, r in after["requests"].items()
                    if rid not in ids
                ), "new request entered RUNNING under gate"
                assert progress, "running requests made no token progress under gate"
                proof["decode_progress_requests"] = len(progress)
                if args.case == "memory":
                    target_blocks = max(2, args.blocks // 2)
                    deadline = time.monotonic() + 9
                    while True:
                        try:
                            resized = await control.request(
                                "POST",
                                "/molink/resize_kv",
                                {"num_blocks": target_blocks},
                            )
                            break
                        except RuntimeError as exc:
                            if (
                                "active request blocks" not in str(exc)
                                or time.monotonic() >= deadline
                            ):
                                raise
                            await asyncio.sleep(0.1)
                    physical = await control.state()
                    proof["released_kv_bytes"] = sum(
                        w["kv_capacity_bytes"] for w in before["workers"]
                    ) - sum(w["kv_capacity_bytes"] for w in physical["workers"])
                    assert proof["released_kv_bytes"] > 0
                    proof["resize"] = resized
                    proof["capacity"] = await control.call_utility_async(
                        "molink_migration_capacity", args.target
                    )
                elif args.case == "failure":
                    await asyncio.sleep(0.3)
                    assert not (
                        await control.call_utility_async("molink_admission_gate")
                    )["enabled"]
                    proof["lease_expiry"] = True
                    # Invalid target exercises executor rejection while a lease exists.
                    await control.call_utility_async(
                        "molink_admission_gate", "open", owner, 2
                    )
                    try:
                        await control.reconfigure(1)
                    except RuntimeError as exc:
                        proof["invalid_migration"] = str(exc)
                    else:
                        raise AssertionError("invalid layer boundary accepted")
                    pending_before = await control.state()
                    cancelled = asyncio.create_task(
                        generate(
                            control,
                            1,
                            args.context,
                            args.tokens,
                            directory / "cancelled_request.jsonl",
                        )
                    )
                    await asyncio.sleep(0.1)
                    pending_after = await control.state()
                    cancelled_ids = set(pending_after["requests"]) - set(
                        pending_before["requests"]
                    )
                    assert cancelled_ids, "cancelled request never reached engine"
                    cancelled.cancel()
                    with suppress(asyncio.CancelledError):
                        await cancelled
                    async with asyncio.timeout(2):
                        while (
                            cancelled_ids & (await control.state())["requests"].keys()
                        ):
                            await asyncio.sleep(0.05)
                    proof["cancelled_requests_removed"] = len(cancelled_ids)
                else:
                    await asyncio.sleep(0.8)
                proof["gate"] = await control.call_utility_async(
                    "molink_admission_gate"
                )
                await control.call_utility_async(
                    "molink_admission_gate", "close", owner
                )
            else:
                path = directory / "model.json"
                shutil.copy2(args.profile, path)
                runtime = AdaptiveScheduler(
                    control,
                    path,
                    directory,
                    OptimizerConfig(
                        stack_size=args.stack,
                        horizon=args.horizon,
                        cooldown=30,
                        admission_control=enabled,
                    ),
                    interval=0.5,
                )
                await asyncio.sleep(2)
                runtime.start()
            await asyncio.sleep(max(0, args.seconds - (time.time() - start)))
        finally:
            if runtime:
                await runtime.stop()
            await control.call_utility_async("molink_admission_gate", "close", owner)
            stop.set()
            await asyncio.gather(*slots)
            await monitoring
        end = time.time()
        final = await control.state()
        append(directory / "final.jsonl", final)
        assert not final["migration_admission_gate"]["enabled"]
        assert final["running_requests"] == final["waiting_requests"] == 0
        mismatches = sum(r["token_ids"] != baseline for r in records)
        assert not mismatches, f"{mismatches} requests have changed token sequences"
        outcome_path = directory / "outcome.log"
        outcomes = (
            [json.loads(line) for line in outcome_path.read_text().splitlines()]
            if outcome_path.exists()
            else []
        )
        successes = [o["result"] for o in outcomes if o["status"] == "resumed"]
        if args.case == "auto":
            assert successes, "no automatic migration completed; inspect decision logs"
            transitions = [
                json.loads(line)["state"]
                for line in (directory / "admission.log").read_text().splitlines()
            ]
            assert "RECLAIMING" in transitions, (
                "automatic migration did not exercise gate"
            )
            assert transitions[-1] == "NORMAL", "admission recovery did not finish"
            proof["transitions"] = transitions
        good = [
            r
            for r in records
            if r["ttft_ms"] <= args.ttft_slo and r["tpot_ms"] <= args.tpot_slo
        ]
        result = {
            "directory": str(directory),
            "case": args.case,
            "concurrency": concurrency,
            "gate_enabled": enabled,
            "initial_boundary": initial["workers"][0]["layer_range"][1],
            "final_boundary": final["workers"][0]["layer_range"][1],
            "repeat": repeat,
            "requests": len(records),
            "elapsed_seconds_including_drain": end - start,
            "ttft_ms": statistics.mean(r["ttft_ms"] for r in records),
            "tpot_ms": statistics.mean(r["tpot_ms"] for r in records),
            "throughput_tokens_s": sum(len(r["token_ids"]) for r in records)
            / (end - start),
            "goodput_tokens_s": sum(len(r["token_ids"]) for r in good) / (end - start),
            "migration_attempts": len(outcomes),
            "migration_successes": len(successes),
            "migration_success_rate": len(successes) / len(outcomes)
            if outcomes
            else None,
            "migration_ms": [r["total_ms"] for r in successes],
            "pause_ms": [r["pause_ms"] for r in successes],
            "gate_seconds": max(
                (
                    s["migration_admission_gate"]["duration_seconds"]
                    for s in samples
                    if s["migration_admission_gate"]["enabled"]
                ),
                default=0,
            ),
            "max_waiting": max((s["waiting_requests"] for s in samples), default=0),
            "min_kv_capacity_bytes": min(
                (sum(w["kv_capacity_bytes"] for w in s["workers"]) for s in samples),
                default=0,
            ),
            "released_kv_bytes": max(
                0,
                sum(w["kv_capacity_bytes"] for w in initial["workers"])
                - min(
                    (
                        sum(w["kv_capacity_bytes"] for w in s["workers"])
                        for s in samples
                    ),
                    default=sum(w["kv_capacity_bytes"] for w in initial["workers"]),
                ),
            ),
            "token_mismatches": mismatches,
            "errors": errors,
            "proof": proof,
        }
        append(args.out / "summary.jsonl", result)
        print(json.dumps(result), flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case", choices=("gate", "memory", "auto", "failure", "comparison")
    )
    parser.add_argument("--url", default="http://127.0.0.1:26030")
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/admission_control/results")
    )
    parser.add_argument(
        "--profile", default="benchmark_results/adaptive-forward/profile.json"
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=192)
    parser.add_argument("--initial-boundary", type=int, default=6)
    parser.add_argument("--target", type=int, default=2)
    parser.add_argument("--stack", type=int, default=2)
    parser.add_argument("--horizon", type=float, default=300)
    parser.add_argument("--delay", type=float, default=3)
    parser.add_argument("--ttft-slo", type=float, default=5000)
    parser.add_argument("--tpot-slo", type=float, default=100)
    args = parser.parse_args()
    for repeat in range(args.repeats):
        for concurrency in args.concurrency:
            for enabled in [False, True] if args.case == "comparison" else [True]:
                await trial(args, concurrency, enabled, repeat)


if __name__ == "__main__":
    asyncio.run(main())
