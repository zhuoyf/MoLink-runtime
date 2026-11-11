# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with torch.distributed.run before serving; two-rank NCCL ping-pong."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
conditions = {
    "gpu_uuids": [str(torch.cuda.get_device_properties(i).uuid) for i in range(2)],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "nccl": list(torch.cuda.nccl.version()),
    "environment": {
        k: v
        for k, v in os.environ.items()
        if k.startswith("NCCL_") or k in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD")
    },
}
if args.check:
    saved = (
        json.loads(Path(args.output).read_text()) if Path(args.output).exists() else {}
    )
    raise SystemExit(0 if saved.get("conditions") == conditions else 1)
rank = int(os.environ["LOCAL_RANK"])
torch.accelerator.set_device_index(rank)
dist.init_process_group("nccl")
rows = []
for size in (1, 4096, 1 << 20, 16 << 20):
    payload = torch.zeros(size, dtype=torch.uint8, device=f"cuda:{rank}")
    samples = []
    for trial in range(15):
        dist.barrier()
        start = time.perf_counter()
        if rank == 0:
            dist.send(payload, 1)
            dist.recv(payload, 1)
        else:
            dist.recv(payload, 0)
            dist.send(payload, 0)
        torch.accelerator.synchronize()
        if trial >= 5:
            samples.append((time.perf_counter() - start) / 2)
    rows.append(
        {"bytes": size, "seconds": sum(samples) / len(samples), "samples": samples}
    )
if rank == 0:
    latency = rows[0]["seconds"]
    bandwidth = rows[-1]["bytes"] / max(1e-9, rows[-1]["seconds"] - latency)
    with open(args.output, "w") as stream:
        json.dump(
            {
                "timestamp": time.time(),
                "conditions": conditions,
                "transport": "nccl",
                "rows": rows,
                "latency_seconds": latency,
                "bytes_per_second": bandwidth,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            stream,
            indent=2,
        )
dist.destroy_process_group()
