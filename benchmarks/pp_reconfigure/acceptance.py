# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assert exact token continuity for live requests across both migration directions."""

import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path


def request(base, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        base + path, data=body, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=180)


def read_json(base, path, data=None):
    with request(base, path, data) as response:
        return json.load(response)


def generation(base, payload, trigger=None, tokens=None):
    output = tokens if tokens is not None else []
    ids = set()
    with request(base, "/v1/completions", {**payload, "stream": True}) as response:
        for line in response:
            if not line.startswith(b"data: ") or b"[DONE]" in line:
                continue
            item = json.loads(line[6:])
            if "error" in item:
                raise RuntimeError(item["error"])
            ids.add(item["id"])
            for choice in item["choices"]:
                output.extend(choice.get("token_ids") or [])
            if trigger is not None and len(output) >= 4:
                trigger.set()
    assert len(ids) == 1, f"request changed identity: {ids}"
    return output


def live_migration(base, payload, boundary, expected):
    ready = threading.Event()
    tokens = []
    errors = []

    def generate():
        try:
            generation(base, payload, ready, tokens)
        except BaseException as exc:
            errors.append(exc)
            ready.set()

    thread = threading.Thread(target=generate, daemon=True)
    thread.start()
    assert ready.wait(60), "request did not start"
    result = read_json(base, "/molink/reconfigure", {"boundary": boundary})
    thread.join(180)
    assert not thread.is_alive(), "request did not resume"
    assert not errors, str(errors)
    assert tokens == expected, f"token mismatch after boundary {boundary}"
    # Logs are the authoritative evidence that live state survived the switch.
    records = [
        json.loads(line)
        for line in (Path(result["log_dir"]) / "state.log").read_text().splitlines()
    ]
    before, after = records[-2:]
    active = {
        key: value
        for key, value in before["requests"].items()
        if value["output_token_ids"]
    }
    assert active, "migration ran without an active decoded request"
    for key, value in active.items():
        assert after["requests"][key] == value, "request state or blocks changed"
        assert value["num_preemptions"] == 0, "request was recomputed"
    assert result["status"] == "resumed"
    receiving = [
        w for w in result["load_and_transfer"] if w["weight_load"]["loaded_bytes"] > 0
    ]
    assert len(receiving) == 1, "destination did not load local layers"
    assert receiving[0]["overlap_ms"] > 0, "weight loading and KV did not overlap"
    return {
        "boundary": boundary,
        "generated_tokens": len(tokens),
        "live_requests": len(active),
        "run_id": result["run_id"],
        "timings_ms": result["timings_ms"],
        "load_and_transfer": result["load_and_transfer"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:26010")
    parser.add_argument("--model", default="molink-tiny")
    parser.add_argument(
        "--output",
        default="/workspace/molink-reconfiguration-results/acceptance/continuity.log",
    )
    args = parser.parse_args()
    payload = {
        "model": args.model,
        "prompt": "The capital of France is " * 8,
        "max_tokens": 128,
        "temperature": 0,
        "ignore_eos": True,
        "seed": 42,
        "return_token_ids": True,
    }
    started = time.perf_counter()
    # Compare identical warm-prefix paths; cold vs cached prefill can round
    # differently on BF16 even before any reconfiguration.
    generation(args.url, {**payload, "max_tokens": 1})
    baseline = generation(args.url, payload)
    assert len(baseline) == payload["max_tokens"]
    results = [
        live_migration(args.url, payload, boundary, baseline) for boundary in (6, 8)
    ]
    monitor = read_json(args.url, "/molink/monitor")
    assert {worker["stage"] for worker in monitor["workers"]} == {0, 1}
    for worker in monitor["workers"]:
        assert "gpu_utilization_percent" in worker, worker
        assert worker["timing_samples"] > 0
    result = {
        "status": "passed",
        "elapsed_s": time.perf_counter() - started,
        "migrations": results,
        "monitor": monitor,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
