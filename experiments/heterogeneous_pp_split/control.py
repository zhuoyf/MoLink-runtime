"""Detached launcher, progress and process-identity-checked cleanup."""

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path("/workspace/results/heterogeneous_pp_split")
PYTHON = str(ROOT / ".venv/bin/python")


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except FileNotFoundError:
        return None


def alive(record):
    return bool(
        record
        and record.get("identity") is not None
        and identity(record["pid"]) == record["identity"]
    )


def read(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def stop_mps():
    result = subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        text=True,
        capture_output=True,
        env={**os.environ, "CUDA_MPS_PIPE_DIRECTORY": str(OUT / "mps/pipe")},
        timeout=30,
    )
    message = result.stdout + result.stderr
    if result.returncode and "Cannot find MPS control daemon process" not in message:
        raise RuntimeError(f"MPS cleanup failed: {message}")


def restore():
    # Only this experiment's recorded process groups and private MPS daemon.
    for record in reversed(read("children.json") or []):
        if not alive(record):
            continue
        with contextlib.suppress(ProcessLookupError):
            os.killpg(record["pid"], signal.SIGTERM)
        end = time.monotonic() + 20
        while alive(record) and time.monotonic() < end:
            time.sleep(0.2)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(record["pid"], signal.SIGKILL)
    stop_mps()
    (OUT / "children.json").write_text("[]\n")
    (OUT / "restored.json").write_text(
        json.dumps(
            {
                "time": time.time(),
                "method": "experiment processes released and private MPS stopped",
            },
            indent=2,
        )
        + "\n"
    )


def supervise(mode):
    lock = (OUT / "background.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = {"pid": os.getpid(), "identity": identity(os.getpid())}
    (OUT / "launcher.json").write_text(json.dumps(record))
    child = subprocess.Popen([PYTHON, str(HERE / "run.py"), mode], cwd=ROOT)

    def stop(*_):
        if child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code = child.wait()
    # Runner holds run.lock for its entire GPU lifetime. Never clean another run.
    run_lock = (OUT / "run.lock").open("w")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    restore()
    status = read("status.json") or {}
    if status.get("status") == "running":
        status.update(status="failed", reason=f"runner exited {code}")
        (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    (OUT / "exit_code").write_text(str(code) + "\n")
    raise SystemExit(code)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "supervise", "status", "stop"])
    p.add_argument("--mode", choices=["search", "smoke"], default="search")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == "supervise":
        supervise(args.mode)
    elif args.action == "start":
        if alive(read("launcher.json")) or alive(read("run.json")):
            raise SystemExit("Experiment already running")
        frozen = read("frozen.json")
        if args.mode == "search" and (not frozen or not frozen["validated"]):
            raise SystemExit("Validated frozen protocol is required")
        (OUT / "exit_code").unlink(missing_ok=True)
        log_path = OUT / (
            "search.log" if args.mode == "search" else "background-smoke.log"
        )
        with log_path.open("a") as log:
            child = subprocess.Popen(
                ["nohup", PYTHON, str(__file__), "supervise", "--mode", args.mode],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"Started {args.mode} supervisor PID {child.pid}; log: {log_path}")
    elif args.action == "status":
        status = read("status.json") or {}
        results = [
            json.loads(p.read_text()) for p in (OUT / "search").glob("*/result.json")
        ]
        print(
            json.dumps(
                {
                    "formal_search_started": bool(results)
                    or status.get("mode") == "search",
                    "frozen_protocol": bool(
                        (read("frozen.json") or {}).get("validated")
                    ),
                    "supervisor_alive": alive(read("launcher.json")),
                    "runner_alive": alive(read("run.json")),
                    "status": status,
                    "progress": read("progress.json"),
                    "finished": len(results),
                    "total": 51,
                    "ok": sum(r["status"] == "ok" for r in results),
                    "failed": sum(r["status"] == "failed" for r in results),
                },
                indent=2,
            )
        )
    else:
        record = read("run.json")
        if alive(record):
            os.kill(record["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 60
            while alive(record) and time.monotonic() < deadline:
                time.sleep(0.5)
            if alive(record):
                os.kill(record["pid"], signal.SIGKILL)
        # The supervisor also performs this recovery if the runner was killed.
        while alive(read("launcher.json")):
            time.sleep(0.5)
        lock = (OUT / "run.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        restore()
        print("Stopped; reservations and private MPS released. GPU clocks unchanged.")


if __name__ == "__main__":
    main()
