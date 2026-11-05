# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify that the master receives periodic NCCL reports without status polling."""

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def read_state(url):
    with urllib.request.urlopen(
        url + "/molink/resource_status", timeout=10
    ) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:26010")
    parser.add_argument(
        "--output",
        default="/workspace/molink-reconfiguration-results/monitoring/resource-sync",
    )
    args = parser.parse_args()
    before = read_state(args.url)
    assert before["transport"] == "nccl" and len(before["workers"]) == 2
    time.sleep(3.2 * before["interval_seconds"])
    after = read_state(args.url)
    master, worker = after["workers"]
    assert after["master_stage"] == 0
    assert master["metrics"]["pid"] == after["master_pid"]
    assert worker["stage"] == 1 and worker["metrics"]["pid"] != after["master_pid"]
    assert master["metrics"]["gpu_uuid"] != worker["metrics"]["gpu_uuid"]
    assert worker["via"] == "nccl" and worker["fresh"]
    received = int(after["reports_received"]["1"]) - int(
        before["reports_received"]["1"]
    )
    assert received >= 2, "reports did not advance while the observer was idle"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    receipts = [
        json.loads(line)
        for line in (output / "master-received.log").read_text().splitlines()
    ]
    receipt = next(
        r
        for r in reversed(receipts)
        if r.get("master_pid") == after["master_pid"]
        and r.get("from_pid") == worker["metrics"]["pid"]
        and r.get("sequence") == worker["metrics"]["sequence"]
    )
    publications = [
        json.loads(line)
        for line in (output / "stage-1-monitor.log").read_text().splitlines()
    ]
    publication = next(
        r
        for r in reversed(publications)
        if r.get("event") == "published"
        and r.get("pid") == receipt["from_pid"]
        and r.get("sequence") == receipt["sequence"]
    )
    assert publication["transport"] == receipt["transport"] == "nccl"
    assert publication["group"] == receipt["group"]
    result = {
        "status": "passed",
        "transport": "nccl",
        "master_stage": 0,
        "master_pid": after["master_pid"],
        "worker_stage": 1,
        "worker_pid": worker["metrics"]["pid"],
        "master_gpu_uuid": master["metrics"]["gpu_uuid"],
        "worker_gpu_uuid": worker["metrics"]["gpu_uuid"],
        "reports_during_observer_idle": received,
        "latest_sequence": worker["metrics"]["sequence"],
        "sample_sequence": worker["metrics"]["sample_sequence"],
        "group": receipt["group"],
        "received_log": receipt,
        "received_at": datetime.fromtimestamp(
            worker["received_time"], ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "snapshot": after,
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = (
        f"PASS: existing PP NCCL group, master stage 0 PID {result['master_pid']}\n"
        f"RECEIVED: stage 1 PID {result['worker_pid']}, "
        f"sequence {result['latest_sequence']}\n"
        f"PERIODIC: {received} new reports while observer made no requests\n"
        f"GPU 0: {result['master_gpu_uuid']}\nGPU 1: {result['worker_gpu_uuid']}\n"
        f"GROUP: {result['group']} | operation=all_gather | transport=nccl\n"
        f"TIME: {result['received_at']}\n"
    )
    (output / "verification.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
