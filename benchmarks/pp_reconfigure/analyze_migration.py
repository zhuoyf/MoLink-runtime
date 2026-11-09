# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize paired runs and verify kernel overlap in an optional CUPTI trace."""

import argparse
import json
import statistics
from pathlib import Path


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def intersection_ms(left, right):
    return sum(
        max(0, min(b, d) - max(a, c)) for a, b in union(left) for c, d in union(right)
    )


def kernel_overlap(path):
    trace = json.loads(Path(path).read_text())
    kernels = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel" and event.get("ph") == "X"
    ]
    migration_streams = {
        event["args"]["stream"] for event in kernels if "_copy_slots" in event["name"]
    }
    compute_streams = {
        event["args"]["stream"]
        for event in kernels
        if "kernel_unified_attention" in event["name"]
    }
    if not migration_streams or not compute_streams:
        raise ValueError("trace lacks migration or attention kernels")
    communication, computation = [], []
    for event in kernels:
        interval = (event["ts"] / 1000, (event["ts"] + event["dur"]) / 1000)
        if (
            "nccl" in event["name"].lower()
            and event["args"]["stream"] in migration_streams
        ):
            communication.append(interval)
        elif (
            event["args"]["stream"] in compute_streams
            and "nccl" not in event["name"].lower()
        ):
            computation.append(interval)
    duration = sum(end - start for start, end in union(communication))
    overlap = intersection_ms(communication, computation)
    if duration <= 0:
        raise ValueError("trace lacks migration NCCL kernels")
    return {
        "trace": str(path),
        "nccl_kernel_ms": duration,
        "inference_kernel_overlap_ms": overlap,
        "kernel_overlap_fraction": overlap / duration,
    }


def main():
    assert intersection_ms([(0, 3), (2, 5)], [(1, 4)]) == 3
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison")
    parser.add_argument("--target", type=int, default=18)
    parser.add_argument("--trace")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.comparison).read_text())
    if data.get("status") != "passed":
        raise ValueError("comparison has not completed successfully")
    rows = [row for row in data["results"] if row["boundary"] == args.target]
    summary = {}
    for mode in ("sync", "async"):
        trials = [row for row in rows if row["mode"] == mode]
        if not trials:
            raise ValueError(f"comparison lacks {mode} trials")
        summary[mode] = {"trials": len(trials)}
        for name, values in {
            "total_ms": [row["total_ms"] for row in trials],
            "pause_ms": [row["pause_ms"] for row in trials],
            "communication_ms": [row["metrics"]["communication_ms"] for row in trials],
            "communication_MiB": [
                row["metrics"]["communication_bytes"] / (1 << 20) for row in trials
            ],
            "overlap_fraction": [row["metrics"]["overlap_fraction"] for row in trials],
        }.items():
            summary[mode][name] = {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
            }
    summary["pause_reduction_fraction"] = (
        1 - summary["async"]["pause_ms"]["mean"] / summary["sync"]["pause_ms"]["mean"]
    )
    if args.trace:
        summary["kernel_profile"] = kernel_overlap(args.trace)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Online pipeline reconfiguration GPU 0 → GPU 1 comparison",
        "",
        "Means of paired runs; complete per-trial data is in the comparison JSON.",
        "",
        "| Mode | Runs | Total ms | Pause ms | NCCL ms | MiB sent | "
        "Forward-span overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("sync", "async"):
        item = summary[mode]
        values = [
            item[key]["mean"]
            for key in ("total_ms", "pause_ms", "communication_ms", "communication_MiB")
        ]
        lines.append(
            f"| {mode} | {item['trials']} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + f" | {item['overlap_fraction']['mean']:.1%} |"
        )
    lines += [
        "",
        f"Pause reduction: {summary['pause_reduction_fraction']:.1%}.",
        "",
        "NCCL ms is source CUDA send-event time, excluding CPU handshake and packing. "
        "Forward-span overlap intersects source forward events with send events; "
        "it includes gaps between inference kernels. A separate CUPTI measurement "
        "uses only actual kernel intervals. Total includes preparation and cleanup. "
        "Pause spans keep-pause/drain through resume. Hash validation is separate "
        "from these performance measurements.",
    ]
    if args.trace:
        item = summary["kernel_profile"]
        lines += [
            "",
            "## Kernel trace",
            "",
            f"Actual inference-kernel overlap: {item['kernel_overlap_fraction']:.1%} "
            f"({item['inference_kernel_overlap_ms']:.3f} / "
            f"{item['nccl_kernel_ms']:.3f} ms). "
            "This is a separate profiled run; profiling overhead is excluded "
            "from the table.",
        ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    modes = ["sync", "async"]
    axes[0].bar(
        modes,
        [summary[mode]["pause_ms"]["mean"] for mode in modes],
        color=["#94a3b8", "#0891b2"],
    )
    axes[0].set_ylabel("Inference pause (ms)")
    axes[1].bar(
        modes,
        [100 * summary[mode]["overlap_fraction"]["mean"] for mode in modes],
        color=["#94a3b8", "#0891b2"],
    )
    axes[1].set_ylabel("Forward-span overlap (%)")
    axes[1].set_ylim(0, 100)
    figure.suptitle("Pipeline reconfiguration: measured migration costs")
    figure.tight_layout()
    figure.savefig(output / "comparison.svg")
    figure.savefig(output / "comparison.png", dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
