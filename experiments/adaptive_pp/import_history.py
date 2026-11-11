# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import measured PipeLive migration costs without mixing model identities."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from vllm.distributed.pp_reconfigure.adaptive.model import PerformanceModel
from vllm.distributed.pp_reconfigure.adaptive.optimizer import PartitionOptimizer

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("output")
args = parser.parse_args()
source = Path(args.source)
data = json.loads(source.read_text())
model = PerformanceModel()
model.conditions = {
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "payload": data["payload"],
    "environment": data.get("server_environment"),
    "note": "Historical migration-only calibration; validate deployment before use.",
}
rows = []
for index, row in enumerate(data["results"]):
    if row.get("mode") != "async" or row.get("status") != "resumed":
        continue
    before = row["state_before"]
    workers = sorted(before["workers"], key=lambda w: w["stage"])
    old = workers[0]["layer_range"][1]
    target = row["boundary"]
    size = row["metrics"]["communication_bytes"]
    key = PartitionOptimizer.migration_key(old, target, size)
    for metric in ("total", "pause"):
        model.update(
            key + "/" + metric, row[metric + "_ms"] / 1000, time.time() + index
        )
    rows.append(
        {
            "boundary": target,
            "old_boundary": old,
            "kv_bytes": size,
            "total_ms": row["total_ms"],
            "pause_ms": row["pause_ms"],
            "service_metrics": row["service_metrics"],
        }
    )
model.conditions["imported_rows"] = len(rows)
model.save(args.output)
Path(args.output).with_suffix(".raw.json").write_text(json.dumps(rows, indent=2))
print(f"Imported {len(rows)} real migrations")
