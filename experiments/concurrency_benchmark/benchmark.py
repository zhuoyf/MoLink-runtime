"""Closed-loop scan; shared simulator and scheduler, isolated server per trial."""

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "heterogeneous_benchmark"))
from benchmark import BASE, PHASES, Control, KVThreshold, configuration
from experiment import append, generate

from vllm.distributed.pp_reconfigure.adaptive.optimizer import OptimizerConfig
from vllm.distributed.pp_reconfigure.adaptive.runtime import AdaptiveScheduler

ROOT = Path(__file__).resolve().parent


def summarize(records, start, end, slo):
    counts = [sum(start <= t < end for t in r["times"]) for r in records]
    good = [
        r["ttft_ms"] <= slo["ttft_ms"] and r["tpot_ms"] <= slo["tpot_ms"]
        for r in records
    ]
    result = dict(
        throughput=sum(counts) / (end - start),
        goodput=sum(n for n, g in zip(counts, good) if g) / (end - start),
        requests=len(records),
        completed_in_window=sum(r["end"] < end for r in records),
    )
    for key in ("tpot_ms", "ttft_ms"):
        for p in (50, 99):
            result[f"{key}_p{p}"] = float(np.percentile([r[key] for r in records], p))
    return result


async def trial(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = configuration(args.seconds)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=1800),
        connector=aiohttp.TCPConnector(limit=0),
    ) as client:
        control = Control(client, "http://127.0.0.1:26020")
        await control.simulate(PHASES["A"])
        warm_start = time.time()
        await generate(control, 4, 32, 16, out / "warmup.jsonl")
        await asyncio.sleep(1.2)
        before = await control.state()
        shutil.copy2(BASE, out / "model.json")
        scheduler = None
        if args.label in ("full", "threshold"):
            scheduler = AdaptiveScheduler(
                control,
                out / "model.json",
                str(out / "decisions"),
                OptimizerConfig(**config["optimizer"]),
            )
            scheduler.not_before = time.time()
            if args.label == "threshold":
                scheduler.optimizer = KVThreshold(scheduler.optimizer.config)
        start = time.time()
        mono = time.monotonic()
        end = start + args.seconds * 3
        deadline = mono + args.seconds * 3
        stop = asyncio.Event()
        active = 0
        records = []

        async def worker(slot):
            nonlocal active
            sequence = 0
            while time.monotonic() < deadline:
                active += 1
                append(
                    out / "load.jsonl",
                    dict(
                        time=time.time(),
                        active=active,
                        event="submit",
                        slot=slot,
                        sequence=sequence,
                    ),
                )
                try:
                    rows = await generate(control, 1, 32, 96, out / "requests.jsonl")
                    records.extend(rows)
                finally:
                    active -= 1
                    append(
                        out / "load.jsonl",
                        dict(
                            time=time.time(),
                            active=active,
                            event="complete",
                            slot=slot,
                            sequence=sequence,
                        ),
                    )
                sequence += 1

        async def sample():
            while not stop.is_set():
                state = await control.state()
                append(
                    out / "samples.jsonl",
                    dict(time=time.time(), client_inflight=active, **state),
                )
                await asyncio.sleep(1)

        async def schedule():
            while time.monotonic() < deadline:
                if scheduler:
                    try:
                        await scheduler.tick()
                    except ValueError as exc:
                        append(
                            out / "scheduler_sample_errors.jsonl",
                            dict(time=time.time(), error=str(exc)),
                        )
                    if scheduler.optimizer.failed:
                        raise RuntimeError("scheduler migration failed")
                await asyncio.sleep(config["decision_interval"])

        async def phases():
            append(out / "phases.jsonl", dict(phase="A", planned=start, time=start))
            for i, phase in enumerate(("B", "C"), 1):
                await asyncio.sleep(max(0, mono + i * args.seconds - time.monotonic()))
                requested = time.time()
                async with control.lock:
                    sim = await control.simulate(PHASES[phase])
                append(
                    out / "phases.jsonl",
                    dict(
                        phase=phase,
                        planned=start + i * args.seconds,
                        requested=requested,
                        time=time.time(),
                        simulation=sim,
                    ),
                )

        async with asyncio.TaskGroup() as group:
            sampling = group.create_task(sample())
            group.create_task(schedule())
            group.create_task(phases())
            workers = [group.create_task(worker(i)) for i in range(args.concurrency)]
            await asyncio.sleep(max(0, deadline - time.monotonic()))
            window_state = await control.state()
            await asyncio.gather(*workers)
            drain_end = time.time()
            stop.set()
            await sampling
        after = await control.state()
        reference = json.loads(
            (ROOT.parent / "heterogeneous_benchmark/pilot/profile/A_16_requests.jsonl")
            .read_text()
            .splitlines()[0]
        )["token_ids"]
        mismatches = sum(r["token_ids"] != reference for r in records)
        summary = dict(
            label=args.label,
            concurrency=args.concurrency,
            start=start,
            end=end,
            duration=end - start,
            warmup_seconds=start - warm_start,
            drain_seconds=drain_end - end,
            before=before,
            window_state=window_state,
            after=after,
            token_mismatches=mismatches,
            **summarize(records, start, end, config["slo"]),
        )
        summary["preemptions"] = max(
            w["preemptions"] for w in window_state["workers"]
        ) - max(w["preemptions"] for w in before["workers"])
        summary["complete"] = not mismatches and all(
            len(r["token_ids"]) == 96 for r in records
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        assert summary["complete"], "token correctness failed"
        print(
            json.dumps(
                {
                    k: v
                    for k, v in summary.items()
                    if k not in ("before", "after", "window_state")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--seconds", type=int, default=200)
    parser.add_argument("--out", required=True)
    asyncio.run(trial(parser.parse_args()))
