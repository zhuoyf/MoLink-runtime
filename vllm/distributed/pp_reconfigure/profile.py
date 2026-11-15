# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrate the adaptive PP model against a running local server."""

import argparse
import asyncio
import hashlib
import json
import time
from contextlib import suppress
from pathlib import Path

import aiohttp

from vllm.distributed.pp_reconfigure.adaptive.model import PerformanceModel
from vllm.distributed.pp_reconfigure.adaptive.optimizer import (
    OptimizerConfig,
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


async def generate(control, batch, context, output):
    async def one():
        payload = {
            "model": control.model_name,
            "prompt": [42] * context,
            "max_tokens": output,
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
        }
        result = await control.request("POST", "/v1/completions", payload)
        if result["usage"]["completion_tokens"] != output:
            raise RuntimeError("calibration request did not complete")

    await asyncio.gather(*(one() for _ in range(batch)))


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
    if (
        path.exists()
        and not args.force
        and PerformanceModel.load(path, conditions).compute_baseline
    ):
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
            task = asyncio.create_task(generate(control, batch, context, 128))
            while not task.done():
                reports = await control.call_utility_async("molink_resource_state")
                reports["model_sample_not_before"] = scheduler.not_before
                append(out / "profile_samples.jsonl", reports)
                with suppress(ValueError):
                    scheduler.ingest(reports, control.version)
                await asyncio.sleep(0.2)
            await task
            # Capture events completed just before the requests finished.
            deadline = time.monotonic() + 10
            while True:
                await asyncio.sleep(0.2)
                reports = await control.call_utility_async("molink_resource_state")
                reports["model_sample_not_before"] = scheduler.not_before
                append(out / "profile_samples.jsonl", reports)
                try:
                    scheduler.ingest(reports, control.version)
                    break
                except ValueError:
                    if time.monotonic() >= deadline:
                        raise
        scheduler.model.save(path)
        print(
            f"Profile boundary {boundary}: {len(scheduler.model.estimates)} buckets",
            flush=True,
        )
    scheduler.model.freeze_compute_baseline()
    scheduler.model.save(path)


async def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as client:
        control = HTTPController(client, args.url.rstrip("/"))
        models = await control.request("GET", "/v1/models")
        control.model_name = models["data"][0]["id"]
        initial = await control.state()
        boundary = next(
            w["layer_range"][1] for w in initial["workers"] if w["stage"] == 0
        )
        try:
            await profile(control, args, out)
        finally:
            current = await control.state()
            if (
                next(w["layer_range"][1] for w in current["workers"] if w["stage"] == 0)
                != boundary
            ):
                await control.reconfigure(boundary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument("--out", default="runtime/profile")
    parser.add_argument("--stack", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
