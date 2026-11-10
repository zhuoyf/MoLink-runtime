# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-GPU multi-request continuity and synchronous/asynchronous comparison."""

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


def generate(base, payload, ready=None, outputs=None, times=None):
    outputs = outputs if outputs is not None else [[] for _ in payload["prompt"]]
    with request(base, "/v1/completions", {**payload, "stream": True}) as response:
        for line in response:
            if not line.startswith(b"data: ") or b"[DONE]" in line:
                continue
            item = json.loads(line[6:])
            if "error" in item:
                raise RuntimeError(item["error"])
            for choice in item["choices"]:
                index = choice["index"]
                ids = choice.get("token_ids") or []
                outputs[index].extend(ids)
                if times is not None and ids:
                    times[index].extend([time.time()] * len(ids))
            if ready is not None and all(len(tokens) >= 4 for tokens in outputs):
                ready.set()
    return outputs


def trial(
    base,
    payload,
    boundary,
    mode,
    expected,
    resize=False,
    verify_kv=False,
):
    ready = threading.Event()
    tokens = [[] for _ in payload["prompt"]]
    times = [[] for _ in tokens]
    errors = []
    request_times = {}

    def run():
        try:
            request_times["submitted"] = time.time()
            generate(base, payload, ready, tokens, times)
            request_times["finished"] = time.time()
        except BaseException as exc:
            errors.append(str(exc))
            ready.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(60), "multi-request generation did not start"
    assert not errors, errors
    resize_results = []
    resize_states = []
    if resize:
        resize_states.append(
            {"time": time.time(), **read_json(base, "/molink/monitor")}
        )
        for capacity in (48, 64):
            resize_results.append(
                read_json(base, "/molink/resize_kv", {"num_blocks": capacity})
            )
            state = {"time": time.time(), **read_json(base, "/molink/monitor")}
            assert state["num_blocks"] == capacity
            assert len(state["requests"]) == len(payload["prompt"])
            assert all(r["num_preemptions"] == 0 for r in state["requests"].values())
            resize_states.append(state)
    migration_started = time.time()
    result = read_json(
        base,
        "/molink/reconfigure",
        {"boundary": boundary, "mode": mode, "verify_kv": verify_kv},
    )
    migration_finished = time.time()
    thread.join(180)
    assert not thread.is_alive() and not errors, errors
    (Path(result["log_dir"]) / "tokens.json").write_text(
        json.dumps({"expected": expected, "actual": tokens}) + "\n"
    )
    assert tokens == expected, (
        f"token continuity failure in {mode} boundary={boundary}; "
        f"logs={result['log_dir']}"
    )
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
    assert len(active) == len(payload["prompt"]), (
        f"expected multiple active requests, found {len(active)}"
    )
    for key, state in active.items():
        assert after["requests"][key] == state
        assert state["num_preemptions"] == 0
    if mode in ("async", "async_no_patching"):
        assert result["metrics"]["inference_steps_during_migration"] > 0
    result["generated_tokens"] = [len(row) for row in tokens]
    result["active_requests_at_commit"] = len(active)
    result["max_stream_gap_ms"] = [
        max((b - a) * 1000 for a, b in zip(row, row[1:])) for row in times
    ]
    result["resize_results"] = resize_results
    result["resize_states"] = resize_states
    result["stream_token_times"] = times
    result["request_times"] = request_times
    result["migration_http_times"] = [migration_started, migration_finished]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:26020")
    parser.add_argument("--model", default="pp-reconfigure-test")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--boundaries", type=int, nargs=2, default=[6, 8])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["sync", "async", "async_no_patching"],
        default=["sync", "async"],
    )
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=192)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--resize-kv", action="store_true")
    parser.add_argument("--verify-kv", action="store_true")
    parser.add_argument(
        "--output",
        default="benchmark_results/pp_reconfigure/comparison.json",
    )
    args = parser.parse_args()
    if (
        min(args.iterations, args.input_tokens, args.output_tokens, args.concurrency)
        < 1
    ):
        parser.error("iterations, token counts and concurrency must be positive")
    if args.output_tokens < 4:
        parser.error("output-tokens must be at least 4 to trigger migration")
    payload = {
        "model": args.model,
        "prompt": [
            [1] + [42 + index] * (args.input_tokens - 1)
            for index in range(args.concurrency)
        ],
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "seed": 42,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    generate(args.url, {**payload, "max_tokens": 1})
    expected = generate(args.url, payload)
    assert all(len(tokens) == payload["max_tokens"] for tokens in expected)
    results = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for iteration in range(args.iterations):
        for mode in args.modes:
            for boundary in args.boundaries:
                result = trial(
                    args.url,
                    payload,
                    boundary,
                    mode,
                    expected,
                    resize=args.resize_kv
                    and iteration == 0
                    and mode == "async"
                    and boundary == args.boundaries[0],
                    verify_kv=args.verify_kv,
                )
                results.append(result)
                output.write_text(json.dumps({"results": results}, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "mode",
                                "boundary",
                                "total_ms",
                                "pause_ms",
                                "metrics",
                                "max_stream_gap_ms",
                            )
                        }
                    ),
                    flush=True,
                )
    output.write_text(
        json.dumps({"status": "passed", "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
