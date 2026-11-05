# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Periodic resource snapshots exchanged over the existing PP NCCL group."""

import json
import math
import os
import socket
import threading
import time
from typing import Any

import torch

from vllm.distributed.pp_reconfigure.monitor import write_event

DEFAULT_LOG_DIR = "/workspace/molink-reconfiguration-results/monitoring/resource-sync"
FRAME_BYTES = 16384


def pack_report(report):
    data = json.dumps(report).encode()
    if len(data) > FRAME_BYTES - 4:
        data = json.dumps(
            {
                "stage": report["stage"],
                "pid": report["pid"],
                "sequence": report["sequence"],
                "sample_error": "resource report exceeds frame capacity",
            }
        ).encode()
    frame = bytearray(FRAME_BYTES)
    frame[:4] = len(data).to_bytes(4, "little")
    frame[4 : 4 + len(data)] = data
    return torch.frombuffer(frame, dtype=torch.uint8)


def unpack_report(frame):
    data = frame.numpy().tobytes()
    size = int.from_bytes(data[:4], "little")
    if not 0 < size <= FRAME_BYTES - 4:
        raise ValueError("invalid resource report length")
    report = json.loads(data[4 : 4 + size])
    for field in ("stage", "pid", "sequence"):
        if type(report.get(field)) is not int:
            raise ValueError(f"{field} must be an integer")
    if report["stage"] not in (0, 1) or report["pid"] <= 0 or report["sequence"] <= 0:
        raise ValueError("invalid resource report identity")
    return report


class ResourceMonitor:
    """Sample independently; publish on the worker main thread in RPC order."""

    def __init__(self, stage, sample, interval=1.0, log_dir=DEFAULT_LOG_DIR):
        if stage not in (0, 1) or not math.isfinite(interval) or interval <= 0:
            raise ValueError("requires stage 0/1 and a finite positive interval")
        self.stage = stage
        self.sample = sample
        self.interval = interval
        self.log_dir = log_dir
        self.sequence = 0
        self.sample_sequence = 0
        self.lock = threading.Lock()
        self.latest: dict[str, Any] = {}
        self.reports: dict[int, dict[str, Any]] = {}
        self.counts = {0: 0, 1: 0}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name=f"MoLinkResourceMonitor-{stage}", daemon=True
        )

    def start(self):
        self.thread.start()

    def sample_once(self):
        self.sample_sequence += 1
        try:
            result = self.sample()
        except Exception as exc:
            result = {"sample_time": time.time(), "sample_error": str(exc)}
        result.update(
            stage=self.stage, pid=os.getpid(), sample_sequence=self.sample_sequence
        )
        with self.lock:
            self.latest = result
        write_event(
            self.log_dir, f"stage-{self.stage}-monitor", event="local_sample", **result
        )

    def _run(self):
        while not self.stop_event.is_set():
            self.sample_once()
            self.stop_event.wait(self.interval)

    def publish(self, group, device):
        """Called by the same collective RPC on both workers, never a timer thread."""
        if group.world_size != 2 or group.rank_in_group != self.stage:
            raise ValueError("resource reporting requires the existing PP2 group")
        self.sequence += 1
        with self.lock:
            report = {
                **self.latest,
                "stage": self.stage,
                "pid": os.getpid(),
                "sequence": self.sequence,
            }
        with torch.cuda.device(device):
            frame = pack_report(report).to(device)
            # GroupCoordinator routes this to its already initialized PyNCCL comm.
            gathered = group.all_gather(frame, dim=0).reshape(2, FRAME_BYTES).cpu()
        write_event(
            self.log_dir,
            f"stage-{self.stage}-monitor",
            event="published",
            transport="nccl",
            operation="all_gather",
            group=group.unique_name,
            **report,
        )
        if self.stage != 0:
            return None
        for frame in gathered:
            incoming = unpack_report(frame)
            stage = incoming["stage"]
            self.reports[stage] = {
                "metrics": incoming,
                "received_time": time.time(),
                "received_monotonic": time.monotonic(),
            }
            self.counts[stage] += 1
            if stage != 0:
                write_event(
                    self.log_dir,
                    "master-received",
                    event="received",
                    master_stage=0,
                    master_pid=os.getpid(),
                    from_stage=stage,
                    from_pid=incoming["pid"],
                    sequence=incoming["sequence"],
                    transport="nccl",
                    operation="all_gather",
                    group=group.unique_name,
                    metrics=incoming,
                )
        return self.snapshot()

    def snapshot(self):
        if self.stage != 0:
            return None
        now = time.monotonic()
        workers = []
        for stage, report in sorted(self.reports.items()):
            age = max(0, now - report["received_monotonic"])
            workers.append(
                {
                    "stage": stage,
                    "metrics": dict(report["metrics"]),
                    "received_time": report["received_time"],
                    "via": "nccl",
                    "age_seconds": age,
                    "fresh": age <= 3 * self.interval,
                }
            )
        return {
            "master_stage": 0,
            "master_pid": os.getpid(),
            "master_host": socket.gethostname(),
            "transport": "nccl",
            "interval_seconds": self.interval,
            "reports_received": dict(self.counts),
            "workers": workers,
            "decision_mode": "manual",
        }

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
