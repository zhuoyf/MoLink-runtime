"""Static PP search. Formal search is an explicit, separate command."""

import argparse
import asyncio
import contextlib
import fcntl
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
from control import identity, stop_mps

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = Path("/workspace/results/heterogeneous_pp_split")
sys.path.insert(0, str(HERE.parent / "adaptive_pp"))
from experiment import HTTPController, append, generate  # noqa: E402

PYTHON = str(ROOT / ".venv/bin/python")
CHILDREN = []
MPS_STARTED = False


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def spawn(cmd, log, env=None):
    with Path(log).open("a") as stream:
        p = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    CHILDREN.append(p)
    save(
        OUT / "children.json",
        [{"pid": x.pid, "identity": identity(x.pid), "cmd": x.args} for x in CHILDREN],
    )
    return p


def terminate(p):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(p.pid, signal.SIGKILL)
    if p in CHILDREN:
        CHILDREN.remove(p)


def start_mps():
    global MPS_STARTED
    directory = OUT / "mps"
    for name in ["pipe", "log"]:
        (directory / name).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(directory / "pipe")
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(directory / "log")
    subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
    MPS_STARTED = True


def cleanup():
    for p in list(reversed(CHILDREN)):
        terminate(p)
    save(OUT / "children.json", [])
    if MPS_STARTED:
        stop_mps()
    # Contention fallback never changes clocks; freeing its processes restores GPUs.
    save(
        OUT / "restored.json",
        {"time": time.time(), "method": "release contention and reservation processes"},
    )


def hardware(config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for gpu in range(2):
        ready = directory / f"gpu{gpu}.json"
        ready.unlink(missing_ok=True)
        p = spawn(
            [
                PYTHON,
                str(HERE / "hardware.py"),
                "hold",
                "--gpu",
                str(gpu),
                "--available",
                str(config["available_gib"][gpu]),
                "--duty",
                str(config["contention_duty"][gpu]),
                "--out",
                str(ready),
            ],
            directory / f"gpu{gpu}.log",
        )
        deadline = time.monotonic() + 60
        while not ready.exists():
            if p.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"GPU {gpu} reservation failed: {directory}")
            time.sleep(0.2)


def calibrate(config):
    directory = OUT / "calibration" / time.strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True)
    result = {
        "method": "CUDA MPS thread cap and fixed-duty real memory copy contention",
        "config": config,
        "clock_control": "nvidia-smi clock changes denied by container permissions",
    }
    for phase in ["baseline", "contended"]:
        if phase == "contended":
            hardware(config, directory / "reservation")
        result[phase] = []
        for gpu in range(2):
            path = directory / f"{phase}-{gpu}.json"
            subprocess.run(
                [
                    PYTHON,
                    str(HERE / "hardware.py"),
                    "bench",
                    "--gpu",
                    str(gpu),
                    "--out",
                    str(path),
                ],
                check=True,
                env={
                    **os.environ,
                    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(
                        config["mps_threads_percent"][gpu]
                        if phase == "contended"
                        else 100
                    ),
                },
            )
            result[phase].append(json.loads(path.read_text()))
    result["ratios_to_own_baseline"] = [
        {
            metric: result["contended"][g][metric] / result["baseline"][g][metric]
            for metric in ["tflops", "gb_s"]
        }
        for g in range(2)
    ]
    save(directory / "calibration.json", result)
    save(OUT / "calibration.json", result)
    print(json.dumps(result["ratios_to_own_baseline"]), flush=True)


def server(config, k, directory):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("VLLM_PP_", "HETVLLM_", "MOLINK_")):
            del env[key]
    env.update(
        CUDA_VISIBLE_DEVICES="0,1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_SERVER_DEV_MODE="1",
        VLLM_PP_RECONFIGURE_ENABLED="1",
        VLLM_PP_RECONFIGURE_ONLINE="0",
        VLLM_PP_LAYER_PARTITION=f"{k},{32 - k}",
        HETVLLM_PP_LINK="0",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        VLLM_MQ_MAX_CHUNK_BYTES_MB="1",
        SPLIT_BUDGET_GIB=json.dumps(config["budget_gib"]),
        SPLIT_MPS_THREADS=json.dumps(config["mps_threads_percent"]),
        PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        LD_PRELOAD="/workspace/molink-reconfiguration-results/pipelive/environment/nccl_shm_redirect.so",
    )
    cmd = [
        str(ROOT / ".venv/bin/vllm"),
        "serve",
        config["model"],
        "--served-model-name",
        "adaptive-test",
        "--host",
        "127.0.0.1",
        "--port",
        str(config["port"]),
        "--pipeline-parallel-size",
        "2",
        "--tensor-parallel-size",
        "1",
        "--distributed-executor-backend",
        "mp",
        "--worker-cls",
        "experiments.heterogeneous_pp_split.worker.BudgetWorker",
        "--dtype",
        "float16",
        "--load-format",
        "safetensors",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(config["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(config["max_num_batched_tokens"]),
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--enforce-eager",
        "--attention-backend",
        "TRITON_ATTN",
    ]
    save(
        directory / "launch.json",
        {
            "command": cmd,
            "environment": {
                k: v
                for k, v in env.items()
                if k.startswith(("VLLM_", "SPLIT_", "HETVLLM_"))
            },
        },
    )
    return spawn(cmd, directory / "server.log", env)


def percentile(values):
    values = sorted(values)
    return (
        values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]
        if values
        else None
    )


def summarize(records, start, end, slo):
    # Fixed window tokens, with drained full requests used to evaluate latency/SLO.
    selected = [r for r in records if r["start"] < end]
    good = [
        r
        for r in selected
        if r["ttft_ms"] <= slo["ttft_ms"] and r["tpot_ms"] <= slo["tpot_ms"]
    ]
    count = lambda rows: sum(sum(start <= t < end for t in r["times"]) for r in rows)
    return {
        "throughput": count(selected) / (end - start),
        "goodput": count(good) / (end - start),
        "p95_ttft_ms": percentile([r["ttft_ms"] for r in selected]),
        "p95_tpot_ms": percentile([r["tpot_ms"] for r in selected]),
        "requests": len(selected),
        "window_seconds": end - start,
        "window_start": start,
        "window_end": end,
        "drain_seconds": max([r["end"] for r in records], default=end) - end,
    }


async def load(control, workload, seconds, directory, config, label):
    records, samples = [], []
    stop = asyncio.Event()
    start = time.time()
    deadline = time.monotonic() + seconds
    end = start + seconds

    async def slot():
        while time.monotonic() < deadline:
            rows = await generate(
                control,
                1,
                workload["input"],
                workload["output"],
                directory / f"{label}-requests.jsonl",
            )
            if len(rows[0]["token_ids"]) != workload["output"]:
                raise RuntimeError("incomplete generation")
            records.extend(rows)

    async def monitor():
        last_progress = 0
        while not stop.is_set():
            elapsed = time.time() - start
            if elapsed - last_progress >= 5 or last_progress == 0:
                save(
                    OUT / "progress.json",
                    {
                        "directory": str(directory),
                        "phase": label,
                        "elapsed_seconds": elapsed,
                        "window_seconds": seconds,
                        "draining": elapsed >= seconds,
                    },
                )
                last_progress = elapsed
            state = await control.state()
            state["time"] = time.time()
            samples.append(state)
            append(directory / f"{label}-samples.jsonl", state)
            for p in CHILDREN:
                if p.poll() is not None:
                    raise RuntimeError(f"child exited: {p.args}")
            await asyncio.sleep(config["sample_seconds"])

    monitoring = asyncio.create_task(monitor())
    slots = [asyncio.create_task(slot()) for _ in range(workload["concurrency"])]
    work = asyncio.gather(*slots)
    try:
        done, _ = await asyncio.wait(
            [work, monitoring], return_when=asyncio.FIRST_COMPLETED
        )
        if monitoring in done:
            await monitoring
        await work
    finally:
        stop.set()
        for task in slots:
            if not task.done():
                task.cancel()
        await asyncio.gather(*slots, return_exceptions=True)
        await monitoring
    result = summarize(records, start, end, config["slo"])
    window = [s for s in samples if start <= s["time"] < end]
    stages = []
    for gpu in range(2):
        workers = [w for s in window for w in s["workers"] if w["stage"] == gpu]
        timings = {
            t["sequence"]: t
            for w in workers
            for t in w["timing_records"]
            if start <= t["timestamp"] < end
        }
        phases = {
            phase: sum(t["gpu_ms"] for t in timings.values() if t["phase"] == phase)
            for phase in ["prefill", "decode"]
        }
        stages.append(
            {
                "stage": gpu,
                "samples": len(workers),
                "phase_gpu_ms": phases,
                "unique_timing_steps": len(timings),
                "forward_busy_fraction": sum(phases.values()) / seconds / 1000,
                "visible_sms": sorted({w["visible_sms"] for w in workers}),
                "stage_gpu_ms_mean": statistics.mean(
                    [t["gpu_ms"] for t in timings.values()]
                )
                if timings
                else None,
                **{
                    key: max([w.get(key, 0) for w in workers], default=0)
                    for key in [
                        "kv_used_bytes",
                        "kv_capacity_bytes",
                        "kv_cache_usage",
                        "torch_peak_allocated_bytes",
                        "torch_peak_reserved_bytes",
                        "device_memory_used_bytes",
                        "preemptions",
                    ]
                },
                "gpu_utilization_mean": statistics.mean(
                    [w.get("gpu_utilization_percent", 0) for w in workers]
                )
                if workers
                else None,
            }
        )
    result["stages"] = stages
    supplied = [s for s in window if s["time"] >= start + 1]
    result["request_supply_active_fraction"] = (
        (
            sum(s["running_requests"] + s["waiting_requests"] > 0 for s in supplied)
            / len(supplied)
        )
        if supplied
        else 0
    )
    return result


async def trial(config, name, k, mode):
    base = OUT / mode / f"{name}-k{k:02d}"
    base.mkdir(parents=True, exist_ok=True)
    directory = base / time.strftime("%Y%m%d-%H%M%S")
    directory.mkdir()
    save(
        directory / "parameters.json", {"config": config, "workload": name, "split": k}
    )
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", config["port"])) == 0:
            raise RuntimeError("server port already occupied")
    print(f"Starting {name} k={k}: {directory}", flush=True)
    save(OUT / "progress.json", {"directory": str(directory), "phase": "startup"})
    p = server(config, k, directory)
    result = {"workload": name, "k": k, "directory": str(directory)}
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config["request_timeout"])
        ) as client:
            control = HTTPController(client, f"http://127.0.0.1:{config['port']}")
            deadline = time.monotonic() + config["startup_timeout"]
            while True:
                if p.poll() is not None:
                    raise RuntimeError(f"server exited {p.returncode}; see server.log")
                try:
                    async with client.get(control.url + "/health") as response:
                        if response.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup timeout")
                await asyncio.sleep(1)
            state = await control.state()
            save(directory / "initial.json", state)
            assert [
                w["layer_range"]
                for w in sorted(state["workers"], key=lambda w: w["stage"])
            ] == [[0, k], [k, 32]]
            assert [
                w["visible_sms"]
                for w in sorted(state["workers"], key=lambda w: w["stage"])
            ] == [84, 48], "MPS cap did not apply"
            assert not (await control.call_utility_async("molink_admission_gate"))[
                "enabled"
            ]
            workload = config["workloads"][name]
            await load(
                control, workload, config["warmup_seconds"], directory, config, "warmup"
            )
            result.update(
                await load(
                    control,
                    workload,
                    config["measure_seconds"],
                    directory,
                    config,
                    "measure",
                )
            )
            final = await control.state()
            save(directory / "final.json", final)
            assert [
                w["layer_range"]
                for w in sorted(final["workers"], key=lambda w: w["stage"])
            ] == [[0, k], [k, 32]]
            assert not (await control.call_utility_async("molink_admission_gate"))[
                "enabled"
            ]
            assert result["requests"] > 0 and result["throughput"] > 0
            result["status"] = "ok"
    except Exception as exc:
        result.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        log = directory / "server.log"
        memory_errors = (
            [
                line
                for line in log.read_text(errors="replace").splitlines()
                if any(
                    word in line.lower()
                    for word in [
                        "out of memory",
                        "no available memory",
                        "maximum number of tokens that can be stored",
                        "less than desired gpu memory",
                    ]
                )
            ]
            if log.exists()
            else []
        )
        result["failure_category"] = (
            "memory_infeasible" if memory_errors else "runtime_error"
        )
        if memory_errors:
            result["reason"] += "\n" + "\n".join(memory_errors[-3:])
    finally:
        terminate(p)
    save(directory / "result.json", result)
    save(base / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def terminal_result(path):
    if not path.exists():
        return False
    return json.loads(path.read_text())["status"] in ("ok", "failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["calibrate", "smoke", "pilot", "search", "report"])
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--smoke-split", type=int, choices=range(8, 25), default=15)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    if args.mode == "report":
        report()
        return
    lock = (OUT / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT / "run.pid").write_text(str(os.getpid()) + "\n")
    save(OUT / "run.json", {"pid": os.getpid(), "identity": identity(os.getpid())})

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    status = "failed"
    try:
        save(
            OUT / "status.json",
            {"status": "running", "mode": args.mode, "pid": os.getpid()},
        )
        if args.mode == "search":
            (OUT / "completed").unlink(missing_ok=True)
        start_mps()
        if args.mode == "calibrate":
            calibrate(config)
        else:
            if args.mode == "search":
                frozen = json.loads((OUT / "frozen.json").read_text())
                if frozen["config"] != config:
                    raise RuntimeError("config differs from validated frozen protocol")
                from freeze import fingerprint

                if frozen["fingerprint"] != fingerprint():
                    raise RuntimeError("experiment code changed since protocol freeze")
                if not frozen["validated"]:
                    raise RuntimeError("pilot has not been validated")
            hardware(config, OUT / "hardware" / time.strftime("%Y%m%d-%H%M%S"))
            if args.mode in ("smoke", "pilot"):
                config = dict(config)
                config.update(
                    warmup_seconds=2, measure_seconds=5 if args.mode == "smoke" else 20
                )
            jobs = (
                [(name, k) for name in config["workloads"] for k in config["splits"]]
                if args.mode == "search"
                else [
                    (n, args.smoke_split if args.mode == "smoke" else 16)
                    for n in (["A"] if args.mode == "smoke" else config["workloads"])
                ]
            )
            for i, (name, k) in enumerate(jobs):
                previous = OUT / args.mode / f"{name}-k{k:02d}" / "result.json"
                if args.mode == "search" and terminal_result(previous):
                    continue
                save(
                    OUT / "status.json",
                    {
                        "status": "running",
                        "mode": args.mode,
                        "pid": os.getpid(),
                        "index": i + 1,
                        "total": len(jobs),
                        "workload": name,
                        "k": k,
                    },
                )
                result = asyncio.run(trial(config, name, k, args.mode))
                if result["status"] != "ok" and args.mode != "search":
                    raise RuntimeError(result["reason"])
            if args.mode == "search":
                report()
        status = "completed"
        if args.mode == "search":
            (OUT / "completed").write_text(str(time.time()) + "\n")
    except KeyboardInterrupt:
        status = "stopped"
    finally:
        cleanup()
        save(
            OUT / "status.json",
            {
                "status": status,
                "mode": args.mode,
                "pid": os.getpid(),
                "time": time.time(),
            },
        )
    if status != "completed":
        raise SystemExit(1)


def report():
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        json.loads(p.read_text())
        for p in sorted((OUT / "search").glob("*/result.json"))
    ]
    keys = [
        "workload",
        "k",
        "status",
        "throughput",
        "goodput",
        "p95_ttft_ms",
        "p95_tpot_ms",
        "reason",
    ]
    with (OUT / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    best = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name in ["A", "B", "C"]:
        group = [r for r in rows if r["workload"] == name and r["status"] == "ok"]
        baseline = next((r for r in group if r["k"] == 16), None)
        best[name] = {}
        for metric in ["throughput", "goodput"]:
            if group:
                winner = max(group, key=lambda r: r[metric])
                best[name][metric] = {
                    "k": winner["k"],
                    "value": winner[metric],
                    "ties": [r["k"] for r in group if r[metric] == winner[metric]],
                    "gain_over_16_percent": 100
                    * (winner[metric] / baseline[metric] - 1)
                    if baseline and baseline[metric]
                    else None,
                }
        for ax, metric in zip(
            axes.flat, ["throughput", "goodput", "p95_ttft_ms", "p95_tpot_ms"]
        ):
            values = {r["k"]: r[metric] for r in group}
            ax.plot(
                range(8, 25),
                [values.get(k, float("nan")) for k in range(8, 25)],
                "o-",
                label=name,
            )
            ax.set(title=metric, xlabel="GPU0 layers", xticks=list(range(8, 25, 2)))
            ax.grid(True)
            ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "curves.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, name in enumerate(["A", "B", "C"]):
        group = [r for r in rows if r["workload"] == name and r["status"] == "ok"]
        for gpu in range(2):
            for ax, metric in [
                (axes[0, j], "stage_gpu_ms_mean"),
                (axes[1, j], "kv_cache_usage"),
            ]:
                values = {r["k"]: r["stages"][gpu][metric] for r in group}
                ax.plot(
                    range(8, 25),
                    [values.get(k, float("nan")) for k in range(8, 25)],
                    "o-",
                    label=f"GPU{gpu}",
                )
                ax.set(title=f"{name}: {metric}", xlabel="GPU0 layers")
                ax.grid(True)
                ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "resources.png")
    plt.close(fig)
    lines = [
        "# 静态 PP 搜索结果",
        "",
        f"已记录 {len(rows)}/51 组；"
        f"失败 {sum(r['status'] == 'failed' for r in rows)} 组。",
        "",
        "最优值按单次测量选择；失败配置不参与排名，并列保留。没有重复实验的统计显著性结论。",
        "",
        "|负载|指标|最优 k|相对 16/16 提升|",
        "|---|---|---|---|",
    ]
    for name, metrics in best.items():
        for metric, value in metrics.items():
            gain = value["gain_over_16_percent"]
            gain_text = f"{gain:.2f}%" if gain is not None else "基线不可用或为零"
            lines.append(f"|{name}|{metric}|{value['ties']}|{gain_text}|")
    for metric in ["throughput", "goodput"]:
        winners = [
            set(best[n][metric]["ties"]) for n in ["A", "B", "C"] if metric in best[n]
        ]
        if len(winners) == 3:
            lines.append(
                f"\n{metric}："
                + (
                    "存在共同最优切分点。"
                    if set.intersection(*winners)
                    else "三类负载没有共同最优切分点。"
                )
            )
    if len(rows) != 51:
        lines.append("\n搜索未完成，当前排名仅覆盖已有结果。")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    save(
        OUT / "best.json",
        {
            "completed_groups": len(rows),
            "expected_groups": 51,
            "best": best,
            "results": rows,
        },
    )


if __name__ == "__main__":
    main()
