# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real HTTP/GPU experiments. Raw responses and samples are retained as JSONL."""

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import time
from contextlib import suppress
from pathlib import Path

import aiohttp

from vllm.distributed.pp_reconfigure.adaptive.model import PerformanceModel
from vllm.distributed.pp_reconfigure.adaptive.optimizer import (
    OptimizerConfig,
    PartitionOptimizer,
)
from vllm.distributed.pp_reconfigure.adaptive.runtime import AdaptiveScheduler


class HTTPController:
    def __init__(self, client, url):
        self.client, self.url = client, url
        self.version = 0
        self.lock = asyncio.Lock()
        self.engine = self
        self.engine_core = self

    async def request(self, method, path, body=None):
        async with self.client.request(method, self.url + path, json=body) as response:
            result = await response.json()
            if response.status != 200:
                raise RuntimeError(f"{response.status}: {result}")
            return result

    async def call_utility_async(self, method, *args):
        if method == "molink_admission_gate":
            if not args:
                return await self.request("GET", "/molink/admission")
            return await self.request(
                "POST",
                "/molink/admission",
                {
                    "action": args[0],
                    "owner": args[1],
                    "timeout": args[2] if len(args) > 2 else 10,
                },
            )
        if method == "molink_migration_capacity":
            return await self.request(
                "GET", f"/molink/migration_capacity?boundary={args[0]}"
            )
        if method != "molink_resource_state":
            raise ValueError(f"unsupported utility: {method}")
        result = await self.request("GET", "/molink/resource_status")
        self.version = result.get("configuration_version", self.version)
        return result

    async def reconfigure(self, boundary, mode="async", **kwargs):
        async with self.lock:
            self.version += 1
            return await self.request(
                "POST",
                "/molink/reconfigure",
                {"boundary": boundary, "mode": mode, **kwargs},
            )

    async def state(self):
        result = await self.request("GET", "/molink/monitor")
        self.version = result.get("configuration_version", self.version)
        return result


def append(path, data):
    with path.open("a") as stream:
        stream.write(json.dumps(data) + "\n")


async def generate(control, batch, context, output, record):
    async def one(index):
        start = time.time()
        times, ids = [], []
        payload = {
            "model": "adaptive-test",
            "prompt": [42] * context,
            "max_tokens": output,
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
            "stream": True,
            "return_token_ids": True,
        }
        async with control.client.post(
            control.url + "/v1/completions", json=payload
        ) as response:
            if response.status != 200:
                raise RuntimeError(await response.text())
            async for line in response.content:
                if not line.startswith(b"data: ") or b"[DONE]" in line:
                    continue
                data = json.loads(line[6:])
                for choice in data.get("choices", []):
                    tokens = choice.get("token_ids") or []
                    if tokens:
                        times.extend([time.time()] * len(tokens))
                        ids.extend(tokens)
        if not times:
            raise RuntimeError("stream did not expose token IDs")
        result = {
            "index": index,
            "start": start,
            "end": time.time(),
            "times": times,
            "token_ids": ids,
            "context": context,
            "batch": batch,
            "ttft_ms": (times[0] - start) * 1000,
            "tpot_ms": (times[-1] - times[0]) * 1000 / max(1, len(times) - 1),
        }
        append(record, result)
        return result

    return await asyncio.gather(*(one(i) for i in range(batch)))


async def profile(control, args, out):
    path = out / "profile.json"
    config = json.loads(Path(args.model, "config.json").read_text())
    state = await control.state()
    conditions = {
        "model": str(Path(args.model).resolve()),
        "config_sha256": hashlib.sha256(
            Path(args.model, "config.json").read_bytes()
        ).hexdigest(),
        "gpu_uuids": [
            w["gpu_uuid"] for w in sorted(state["workers"], key=lambda w: w["stage"])
        ],
        "activation_bytes_per_token": config["hidden_size"] * 2,
        "total_layers": config["num_hidden_layers"],
        "block_size": state["workers"][0].get("block_size", 16),
        "kv_payload_bytes_per_layer_block": state["workers"][0].get(
            "kv_payload_bytes_per_layer_block"
        ),
        "stack_size": args.stack,
        "dtype": "bfloat16",
        "compute_timing_scope": state["workers"][0].get("compute_timing_scope"),
        "source": "live_profile",
        "network_conditions": json.loads((out / "network.json").read_text()).get(
            "conditions"
        ),
    }
    if path.exists() and not args.force:
        PerformanceModel.load(path, conditions)
        print("Reusing matching profile", flush=True)
        return
    model = PerformanceModel(args.alpha)
    model.conditions = conditions
    network = json.loads((out / "network.json").read_text())
    for metric in ("latency_seconds", "bytes_per_second"):
        model.update("network/" + metric, network[metric], network["timestamp"])
    model.save(path)
    scheduler = AdaptiveScheduler(
        control,
        path,
        str(out / "profile_decisions"),
        OptimizerConfig(stack_size=args.stack),
    )
    scheduler.not_before = time.time()
    boundaries = list(range(args.stack, config["num_hidden_layers"], args.stack))
    for boundary in boundaries + boundaries[::-1]:
        current = await control.state()
        old = current["workers"][0]["layer_range"][1]
        if boundary != old:
            result = await control.reconfigure(boundary)
            scheduler.not_before = time.time()
            append(
                out / "profile_migrations.jsonl", {"before": current, "result": result}
            )
            source = next(
                w
                for w in current["workers"]
                if w["stage"] == (1 if boundary > old else 0)
            )
            layers = source["layer_range"][1] - source["layer_range"][0]
            page_bytes = source.get("kv_payload_bytes_per_layer_block")
            size = (
                abs(boundary - old) * page_bytes * source["active_kv_blocks"]
                if page_bytes is not None
                else abs(boundary - old) * source["kv_used_bytes"] / layers
            )
            key = scheduler.optimizer.migration_key(old, boundary, size)
            for metric in ("total", "pause"):
                scheduler.model.update(
                    key + "/" + metric, result[metric + "_ms"] / 1000, time.time()
                )
            # Idle migrations use a conservative service-interference bound.
            scheduler.model.update(key + "/interference", 1.0, time.time())
        for batch, context in ((1, 32), (4, 32), (4, 128)):
            task = asyncio.create_task(
                generate(control, batch, context, 128, out / "profile_requests.jsonl")
            )
            while not task.done():
                reports = await control.call_utility_async("molink_resource_state")
                reports["model_sample_not_before"] = scheduler.not_before
                append(out / "profile_samples.jsonl", reports)
                with suppress(ValueError):
                    scheduler.ingest(reports, control.version)
                await asyncio.sleep(0.2)
            await task
            # Capture events completed just before the requests finished.
            await asyncio.sleep(1.1)
            reports = await control.call_utility_async("molink_resource_state")
            reports["model_sample_not_before"] = scheduler.not_before
            append(out / "profile_samples.jsonl", reports)
            scheduler.ingest(reports, control.version)
        scheduler.model.save(path)
        print(
            f"Profile boundary {boundary}: {len(scheduler.model.estimates)} buckets",
            flush=True,
        )
    scheduler.model.freeze_compute_baseline()
    scheduler.model.save(path)


async def workload(control, scheduler, args, out, label):
    records = []
    stop = asyncio.Event()

    async def sample():
        next_decision = 0
        while not stop.is_set():
            try:
                reports = await control.call_utility_async("molink_resource_state")
                append(out / (label + "_samples.jsonl"), reports)
                if scheduler and time.monotonic() >= next_decision:
                    await scheduler.tick()
                    next_decision = time.monotonic() + 5
            except Exception as exc:
                append(
                    out / (label + "_errors.jsonl"),
                    {"error": str(exc), "time": time.time()},
                )
            await asyncio.sleep(1)

    task = asyncio.create_task(sample())
    try:
        started = time.monotonic()
        waves = []
        for offset, batch, context in ((0, 1, 32), (3, 4, 128), (6, 8, 64), (9, 1, 32)):
            await asyncio.sleep(max(0, started + offset - time.monotonic()))
            waves.append(
                asyncio.create_task(
                    generate(
                        control,
                        batch,
                        context,
                        args.output_tokens,
                        out / (label + "_requests.jsonl"),
                    )
                )
            )
        for wave in await asyncio.gather(*waves):
            records.extend(wave)
        if args.experiment == "online_update":
            # A stationary tail distinguishes convergence from workload changes.
            for _ in range(8):
                records.extend(
                    await generate(
                        control,
                        4,
                        32,
                        args.output_tokens,
                        out / (label + "_requests.jsonl"),
                    )
                )
    finally:
        stop.set()
        await task
    duration = max(r["end"] for r in records) - min(r["start"] for r in records)
    tokens = sum(len(r["token_ids"]) for r in records)
    good = [r for r in records if r["ttft_ms"] <= 5000 and r["tpot_ms"] <= 150]
    summary = {
        "label": label,
        "throughput_tokens_s": tokens / duration,
        "slo_goodput_tokens_s": sum(len(r["token_ids"]) for r in good) / duration,
        "mean_tpot_ms": sum(r["tpot_ms"] for r in records) / len(records),
        "duration": duration,
        "requests": len(records),
        "generated_tokens": tokens,
    }
    append(out / "summary.jsonl", summary)
    return summary


class ThresholdOptimizer(PartitionOptimizer):
    def decide(self, state, model, now):
        result = super().decide(state, model, now)
        result["target"] = state.boundary
        if result["reason"] in (
            "cooldown",
            "failure_latched",
            "paused_or_stale",
            "missing_nodes",
        ):
            return result
        pressure = max(n.used_blocks / max(1, n.num_blocks - 1) for n in state.nodes)
        difference = state.nodes[0].gpu_utilization - state.nodes[1].gpu_utilization
        if pressure < 0.8 and abs(difference) < 20:
            result["reason"] = "below_threshold"
            return result
        target = state.boundary + (
            -self.config.stack_size if difference > 0 else self.config.stack_size
        )
        if any(c["boundary"] == target and c["valid"] for c in result["candidates"]):
            result.update(target=target, reason="resource_threshold")
        return result


def archive_previous_run(out, experiment):
    """Keep repeated experiments separate while retaining their raw records."""
    paths = [
        p
        for p in out.iterdir()
        if p.name != "scheduler_report.md"
        and (
            p.name == experiment
            or p.name.startswith(experiment + "_")
            or p.name.startswith(experiment + ".")
        )
    ]
    if not paths:
        return
    archive = out / "archive" / f"{experiment}-{time.time_ns()}"
    archive.mkdir(parents=True)
    for path in paths:
        shutil.move(str(path), archive / path.name)
    summary_path = out / "summary.jsonl"
    if summary_path.exists():
        retained = []
        for line in summary_path.read_text().splitlines():
            row = json.loads(line)
            if row["label"] == experiment or row["label"].startswith(experiment + "_"):
                append(archive / "summary.jsonl", row)
            else:
                retained.append(line)
        summary_path.write_text("\n".join(retained) + ("\n" if retained else ""))


async def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.experiment != "profile":
        archive_previous_run(out, args.experiment)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as client:
        control = HTTPController(client, args.url)
        if args.experiment == "profile":
            await profile(control, args, out)
        elif args.server_managed:
            if args.experiment != "scheduler":
                raise ValueError(
                    "server-managed verification requires scheduler experiment"
                )
            before = await control.request("GET", "/molink/adaptive/status")
            if not before["running"]:
                raise RuntimeError("server adaptive scheduler is not running")
            summary = await workload(control, None, args, out, "scheduler")
            after = await control.request("GET", "/molink/adaptive/status")
            if (
                not after["running"]
                or after["failure_latched"]
                or after["latest"] is None
            ):
                raise RuntimeError("server scheduler failed to produce a live decision")
            (out / "startup_status.json").write_text(
                json.dumps({"before": before, "after": after}, indent=2)
            )
            print(json.dumps(summary), flush=True)
        elif args.experiment == "comparison":
            config = json.loads(Path(args.model, "config.json").read_text())
            boundaries = list(
                range(args.stack, config["num_hidden_layers"], args.stack)
            )
            summaries = []
            for repeat in range(args.repeats):
                labels = [(f"static_{b}", b) for b in boundaries] + [
                    ("threshold", boundaries[-1]),
                    ("full", boundaries[-1]),
                ]
                if repeat % 2:
                    labels.reverse()
                for label, boundary in labels:
                    current = await control.state()
                    if current["workers"][0]["layer_range"][1] != boundary:
                        await control.reconfigure(boundary)
                    await asyncio.sleep(1.2)
                    await control.request("POST", "/reset_prefix_cache")
                    name = f"comparison_{repeat}_{label}"
                    await generate(control, 1, 32, 16, out / "warmup_requests.jsonl")
                    scheduler = None
                    if label in ("threshold", "full"):
                        profile_path = out / (name + ".model.json")
                        shutil.copy2(out / "profile.json", profile_path)
                        scheduler = AdaptiveScheduler(
                            control,
                            profile_path,
                            str(out / name),
                            OptimizerConfig(stack_size=args.stack),
                        )
                        if label == "threshold":
                            scheduler.optimizer = ThresholdOptimizer(
                                OptimizerConfig(stack_size=args.stack)
                            )
                    summaries.append(
                        await workload(control, scheduler, args, out, name)
                    )
            (out / "comparison.json").write_text(json.dumps(summaries, indent=2))
        else:
            shutil.copy2(out / "profile.json", out / (args.experiment + ".model.json"))
            scheduler = AdaptiveScheduler(
                control,
                out / (args.experiment + ".model.json"),
                str(out / args.experiment),
                OptimizerConfig(stack_size=args.stack, cooldown=10),
            )
            if args.experiment == "online_update":
                scheduler.optimizer.config.min_gain = 0.99
            contender = None
            if args.contention:
                state = await control.state()
                target = (
                    sum(
                        w["layer_range"][1] - w["layer_range"][0]
                        for w in state["workers"]
                    )
                    - args.stack
                )
                if state["workers"][0]["layer_range"][1] != target:
                    await control.reconfigure(target)
                contender = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(Path(__file__).with_name("contention.py")),
                    "--seconds",
                    "20",
                    "--gpu",
                    "0",
                )
                await asyncio.sleep(3)
            try:
                summary = await workload(control, scheduler, args, out, args.experiment)
                print(json.dumps(summary), flush=True)
            finally:
                if contender:
                    await contender.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment", choices=["profile", "online_update", "scheduler", "comparison"]
    )
    parser.add_argument("--url", default="http://127.0.0.1:26020")
    parser.add_argument("--out", default="benchmark_results/adaptive")
    parser.add_argument("--model", default="/workspace/model/molink-tiny-llama")
    parser.add_argument("--stack", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--contention", action="store_true")
    parser.add_argument("--server-managed", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    asyncio.run(main(parser.parse_args()))
