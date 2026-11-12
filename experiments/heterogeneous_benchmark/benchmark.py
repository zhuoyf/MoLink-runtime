# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-trace heterogeneous PP experiments; simulation affects real execution."""

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive_pp"))
from experiment import HTTPController, append, generate  # noqa: E402

from vllm.distributed.pp_reconfigure.adaptive.optimizer import (
    OptimizerConfig,
    PartitionOptimizer,
)
from vllm.distributed.pp_reconfigure.adaptive.runtime import AdaptiveScheduler

ROOT = Path(__file__).resolve().parent
BASE = Path("benchmark_results/adaptive-final-llama7b/profile.json")
PHASES = {
    "A": [{"decode_ms_per_layer": 0, "kv_budget_bytes": 2 << 30}] * 2,
    "B": [
        {"decode_ms_per_layer": 0, "kv_budget_bytes": 2 << 30},
        {"decode_ms_per_layer": 5, "kv_budget_bytes": 768 << 20},
    ],
    "C": [
        {"decode_ms_per_layer": 5, "kv_budget_bytes": 768 << 20},
        {"decode_ms_per_layer": 0, "kv_budget_bytes": 2 << 30},
    ],
}


class Control(HTTPController):
    stages = PHASES["A"]

    async def simulate(self, stages, target=None):
        result = await self.request(
            "POST",
            "/molink/heterogeneity",
            {"stages": stages, "target_boundary": target},
        )
        self.stages = stages
        self.version = result["configuration_version"]
        for w in result["workers"]:
            assert w["num_blocks"] == result["capacity"]
            assert w["kv_capacity_bytes"] <= stages[w["stage"]]["kv_budget_bytes"]
        return result

    async def reconfigure(self, boundary, mode="async", **kwargs):
        async with self.lock:
            current = await self.state()
            old = current["workers"][0]["layer_range"][1]
            if old == boundary:
                return None
            if kwargs.get("expected_version", self.version) != self.version:
                raise RuntimeError("configuration changed before migration")
            transaction_started = time.time()
            await self.simulate(self.stages, boundary)
            result = await self.request(
                "POST",
                "/molink/reconfigure",
                {
                    "boundary": boundary,
                    "mode": mode,
                    "verify_kv": True,
                    "expected_version": self.version,
                    "expected_boundary": old,
                },
            )
            await self.simulate(self.stages)
            result["pipelive_total_ms"] = result["total_ms"]
            result["total_ms"] = (time.time() - transaction_started) * 1000
            result["simulation_resize_ms"] = (
                result["total_ms"] - result["pipelive_total_ms"]
            )
            return result


class KVThreshold(PartitionOptimizer):
    def decide(self, state, model, now):
        result = super().decide(state, model, now)
        result["target"] = state.boundary
        if result["reason"] in (
            "cooldown",
            "failure_latched",
            "paused_or_stale",
            "idle",
        ):
            return result
        pressure = [
            n.used_blocks
            / max(
                1,
                min(
                    n.block_cap or n.num_blocks,
                    int(
                        (n.kv_budget_bytes or 1e30)
                        / (n.layers * n.kv_bytes_per_layer_block)
                    ),
                )
                - 1,
            )
            for n in state.nodes
        ]
        result["threshold_pressure"] = pressure
        target = state.boundary + (4 if pressure[1] > pressure[0] else -4)
        result["reason"] = "below_kv_threshold"
        if max(pressure) >= 0.6 and any(
            c["boundary"] == target and c["valid"] for c in result["candidates"]
        ):
            result.update(target=target, reason="kv_threshold")
        return result


async def reset(control, boundary):
    await control.simulate(PHASES["A"])
    await control.reconfigure(boundary)
    await control.request("POST", "/reset_prefix_cache")
    await generate(control, 4, 32, 16, ROOT / "logs/warmup.jsonl")
    await asyncio.sleep(1.2)


def metrics(records, start, end, slo):
    good = [
        r
        for r in records
        if r["ttft_ms"] <= slo["ttft_ms"] and r["tpot_ms"] <= slo["tpot_ms"]
    ]
    return {
        "throughput": sum(len(r["token_ids"]) for r in records) / (end - start),
        "goodput": sum(len(r["token_ids"]) for r in good) / (end - start),
        "tpot_ms": sum(r["tpot_ms"] for r in records) / len(records),
        "ttft_ms": sum(r["ttft_ms"] for r in records) / len(records),
        "latency_ms": sum((r["end"] - r["start"]) * 1000 for r in records)
        / len(records),
        "requests": len(records),
        "duration": end - start,
    }


async def profile(control):
    out = ROOT / "pilot/profile"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for phase, stages in PHASES.items():
        for boundary in range(4, 32, 4):
            name = f"{phase}_{boundary}"
            path = out / f"{name}.json"
            if path.exists():
                results.append(json.loads(path.read_text()))
                continue
            await reset(control, boundary)
            sim = await control.simulate(stages)
            started = time.time()
            records = await generate(control, 4, 32, 96, out / f"{name}_requests.jsonl")
            state = await control.state()
            result = {
                "phase": phase,
                "boundary": boundary,
                "simulation": sim,
                "state": state,
                **metrics(
                    records, started, time.time(), {"ttft_ms": 5000, "tpot_ms": 100}
                ),
            }
            path.write_text(json.dumps(result, indent=2))
            results.append(result)
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "phase",
                            "boundary",
                            "throughput",
                            "goodput",
                            "tpot_ms",
                        )
                    }
                ),
                flush=True,
            )
    best = {
        p: max(
            [r for r in results if r["phase"] == p],
            key=lambda r: (r["goodput"], r["throughput"]),
        )["boundary"]
        for p in PHASES
    }
    assert best["B"] != best["A"] and best["C"] != best["B"], best
    (ROOT / "pilot/phase_best.json").write_text(json.dumps(best, indent=2))
    return best


async def run_trial(control, label, config, out, oracle=None):
    out.mkdir(parents=True, exist_ok=False)
    boundary = (
        int(label.split("_")[-1])
        if label.startswith("static_")
        else config["initial_boundary"]
    )
    if label == "full_start28":
        boundary = 28
    await reset(control, boundary)
    before = await control.state()
    shutil.copy2(BASE, out / "model.json")
    scheduler = None
    if label in (
        "full",
        "full_start28",
        "threshold",
        "compute_only",
        "without_compute",
        "without_migration_cost",
    ):
        scheduler = AdaptiveScheduler(
            control,
            out / "model.json",
            str(out / "decisions"),
            OptimizerConfig(**config["optimizer"]),
        )
        scheduler.not_before = time.time()
        if label == "threshold":
            scheduler.optimizer = KVThreshold(scheduler.optimizer.config)
        if label == "compute_only":
            original = scheduler.optimizer.evaluate

            def evaluate(state, model, boundary, now):
                result = original(state, model, boundary, now)
                if result.get("valid"):
                    result["risk"] = 0
                return result

            scheduler.optimizer.evaluate = evaluate
        if label == "without_compute":
            original = scheduler.optimizer.evaluate

            def evaluate(state, model, boundary, now):
                result = original(state, model, boundary, now)
                if result.get("valid"):
                    result["step_seconds"] = 1.0
                return result

            scheduler.optimizer.evaluate = evaluate
        if label == "without_migration_cost":
            original = scheduler.optimizer.evaluate

            def evaluate(state, model, boundary, now):
                result = original(state, model, boundary, now)
                if result.get("valid"):
                    result.update(migration_total=0, migration_pause=0, interference=0)
                return result

            scheduler.optimizer.evaluate = evaluate
    stop = asyncio.Event()

    async def sample():
        next_tick = 0
        while not stop.is_set():
            try:
                state = await control.state()
                append(out / "samples.jsonl", {"time": time.time(), **state})
                if scheduler and time.monotonic() >= next_tick:
                    await scheduler.tick()
                    next_tick = time.monotonic() + config["decision_interval"]
            except Exception as exc:
                append(out / "errors.jsonl", {"time": time.time(), "error": str(exc)})
                if scheduler and scheduler.optimizer.failed:
                    raise
            await asyncio.sleep(1)

    task = asyncio.create_task(sample())
    start = time.time()
    phase_seconds = config["phase_seconds"]

    async def phases():
        for i, p in enumerate(PHASES):
            await asyncio.sleep(max(0, start + i * phase_seconds - time.time()))
            async with control.lock:
                sim = await control.simulate(config["phases"][p])
            append(
                out / "phases.jsonl",
                {
                    "phase": p,
                    "planned": start + i * phase_seconds,
                    "time": time.time(),
                    "simulation": sim,
                },
            )
            if oracle:
                result = await control.reconfigure(oracle[p])
                append(
                    out / "oracle_migrations.jsonl",
                    {"time": time.time(), "phase": p, "result": result},
                )

    phase_task = asyncio.create_task(phases())
    waves = []
    try:
        for index, arrival in enumerate(config["arrivals"]):
            await asyncio.sleep(max(0, start + arrival["offset"] - time.time()))
            waves.append(
                asyncio.create_task(
                    generate(
                        control,
                        arrival["batch"],
                        arrival["context"],
                        arrival["output"],
                        out / "requests.jsonl",
                    )
                )
            )
        records = [r for wave in await asyncio.gather(*waves) for r in wave]
        await phase_task
        await asyncio.sleep(max(0, start + 3 * phase_seconds - time.time()))
        end = time.time()
    finally:
        stop.set()
        await task
    after = await control.state()
    counters = []
    for a, b in zip(before["workers"], after["workers"]):
        counters.append(
            {
                k: b["simulation_counters"][k] - a["simulation_counters"][k]
                for k in a["simulation_counters"]
            }
        )
    summary = {
        "label": label,
        "start": start,
        "end": end,
        "initial_boundary": boundary,
        "counters": counters,
        "before": before,
        "after": after,
        **metrics(records, start, end, config["slo"]),
    }
    summary["complete"] = all(
        len(r["token_ids"]) == config["arrivals"][0]["output"] for r in records
    )
    summary["preemptions"] = max(w["preemptions"] for w in after["workers"]) - max(
        w["preemptions"] for w in before["workers"]
    )
    reference = json.loads(
        (ROOT / "pilot/profile/A_16_requests.jsonl").read_text().splitlines()[0]
    )["token_ids"]
    summary["token_mismatches"] = sum(r["token_ids"] != reference for r in records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    assert summary["complete"] and summary["token_mismatches"] == 0
    assert summary["preemptions"] == 0
    assert all(
        c["resumed_requests"] == 0 and c["prefill_tokens"] == 32 * len(records)
        for c in counters
    )
    if scheduler and scheduler.optimizer.failed:
        raise RuntimeError("scheduler failed")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "label",
                    "goodput",
                    "throughput",
                    "tpot_ms",
                    "complete",
                    "preemptions",
                )
            }
        ),
        flush=True,
    )
    return summary


def configuration(seconds):
    return {
        "phases": PHASES,
        "phase_seconds": seconds,
        "initial_boundary": 16,
        "repeats": 5,
        "slo": {"ttft_ms": 5000, "tpot_ms": 100},
        "decision_interval": 5,
        "optimizer": {"stack_size": 4, "horizon": 90, "cooldown": 20, "min_gain": 0.05},
        "arrivals": [
            {"offset": offset, "batch": 4, "context": 32, "output": 96}
            for offset in range(0, 3 * seconds, 12)
        ],
        "oracle": "phase-aware, actual migration costs included",
        "profile_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
    }


async def main(args):
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=3600),
        connector=aiohttp.TCPConnector(force_close=True),
    ) as client:
        control = Control(client, args.url)
        if args.mode == "profile":
            await profile(control)
        elif args.mode == "pilot":
            await run_trial(control, "full", configuration(60), ROOT / "pilot/dynamic")
        elif args.mode == "formal":
            config = json.loads((ROOT / "frozen.json").read_text())
            assert (
                hashlib.sha256(BASE.read_bytes()).hexdigest()
                == config["profile_sha256"]
            )
            best = json.loads((ROOT / "pilot/phase_best.json").read_text())
            labels = [f"static_{b}" for b in range(4, 32, 4)] + [
                "threshold",
                "full",
                "oracle",
                "compute_only",
                "without_compute",
                "full_start28",
                "without_migration_cost",
            ]
            for repeat in range(config["repeats"]):
                order = labels[repeat:] + labels[:repeat]
                if repeat % 2:
                    order.reverse()
                for label in order:
                    out = ROOT / f"results/{repeat}_{label}"
                    print(f"START repeat={repeat} strategy={label}", flush=True)
                    await run_trial(
                        control, label, config, out, best if label == "oracle" else None
                    )
        else:
            raise ValueError(args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["profile", "pilot", "formal"])
    parser.add_argument("--url", default="http://127.0.0.1:26020")
    asyncio.run(main(parser.parse_args()))
