# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot head phase timings and receiving-worker overlap from migration logs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASES = [
    ("pause", "Pause / drain", "#64748b"),
    ("prepare", "Prepare buffers / layers", "#d49b36"),
    ("connect", "NIXL connect", "#94a3b8"),
    ("load_and_transfer", "Weight load + KV", "#345ca8"),
    ("commit", "Commit", "#8870b5"),
    ("finalize", "Release old state", "#538978"),
    ("resume", "Resume", "#219a80"),
]


def head_interval(directory, phase):
    records = [
        json.loads(line)
        for line in (directory / f"{phase}.log").read_text().splitlines()
    ]
    start = next(r for r in records if r.get("event") == "begin")
    end = next(r for r in records if r.get("event") == "end")
    return start["time"], end["time"], end["elapsed_ms"]


def plot(report_path, output):
    report = json.loads(report_path.read_text())
    assert report["status"] == "passed"
    assert len(report["migrations"]) == 2
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(
        2, 2, figsize=(15, 9), gridspec_kw={"height_ratios": [2.2, 1.4]}
    )
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(
        "Pipeline reconfiguration: measured synchronous migration",
        fontsize=21,
        fontweight="bold",
        y=0.98,
        color="#172c48",
    )
    fig.text(
        0.5,
        0.932,
        "Token continuity and measured migration phases",
        ha="center",
        color="#52647a",
        fontsize=11,
    )

    for column, migration in enumerate(report["migrations"]):
        directory = report_path.parent.parent / "migrations" / migration["run_id"]
        target = next(
            w
            for w in migration["load_and_transfer"]
            if w["weight_load"]["loaded_bytes"] > 0
        )
        boundary = migration["boundary"]
        previous = 8 if boundary == 6 else 6
        title = f"Stage 0: {previous} -> {boundary} layers"
        ax = axes[0, column]
        intervals = [head_interval(directory, name) for name, _, _ in PHASES]
        origin = intervals[0][0]
        total = (intervals[-1][1] - origin) * 1000
        for row, ((_, label, color), (start, end, elapsed)) in enumerate(
            zip(PHASES, intervals)
        ):
            ax.barh(
                row,
                (end - start) * 1000,
                left=(start - origin) * 1000,
                height=0.55,
                color=color,
            )
            ax.text(
                (end - origin) * 1000 + total * 0.012,
                row,
                f"{elapsed:.2f} ms",
                va="center",
                fontsize=9,
                color="#334155",
            )
        ax.set_yticks(range(len(PHASES)), [p[1] for p in PHASES])
        ax.invert_yaxis()
        ax.set_xlim(0, total * 1.16)
        ax.set_xlabel("Elapsed since pause began (ms)")
        ax.set_title(
            f"{title}\nHead timeline: {total:.1f} ms",
            loc="left",
            fontsize=13,
            fontweight="bold",
            pad=14,
        )

        ax = axes[1, column]
        weight, kv = target["weight_load"], target["kv"]
        origin = min(weight["started_ns"], kv["started_ns"])
        overlap_start = (max(weight["started_ns"], kv["started_ns"]) - origin) / 1e6
        overlap_end = (min(weight["finished_ns"], kv["finished_ns"]) - origin) / 1e6
        assert overlap_end > overlap_start
        ax.axvspan(overlap_start, overlap_end, color="#1fae91", alpha=0.12)
        for row, (record, color) in enumerate(((kv, "#169c84"), (weight, "#345ca8"))):
            left = (record["started_ns"] - origin) / 1e6
            width = (record["finished_ns"] - record["started_ns"]) / 1e6
            ax.barh(row, width, left=left, height=0.36, color=color)
            ax.text(
                left + width + 10,
                row,
                f"{record['elapsed_ms']:.2f} ms",
                va="center",
                fontsize=10,
                color="#334155",
            )
        ax.set_yticks([0, 1], ["KV transfer (64 MiB)", "Local weight load (810 MB)"])
        ax.set_ylim(-0.65, 1.65)
        ax.set_xlim(0, 910)
        ax.set_xlabel("Elapsed within receiving worker (ms)")
        ax.set_title(
            f"Receiver stage {target['stage']}: overlap {target['overlap_ms']:.2f} ms",
            loc="left",
            fontsize=13,
            fontweight="bold",
            pad=14,
        )
        ax.text(
            0.02,
            0.04,
            "Entire KV interval overlaps local loading",
            transform=ax.transAxes,
            fontsize=9,
            color="#167d69",
        )

    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.xaxis.grid(True, color="#e3e9ef", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, labelcolor="#46576c")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(
        0.02,
        0.025,
        "Prepare includes first-use NIXL setup. Head panels use wall-clock events; "
        "receiver panels use per-worker monotonic timestamps.\n"
        f"Source: {report_path.name} and migrations/<run_id>/*.log",
        fontsize=9,
        color="#64748b",
    )
    fig.subplots_adjust(
        top=0.85, bottom=0.13, left=0.17, right=0.97, hspace=0.69, wspace=0.57
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            output.with_suffix(f".{extension}"),
            dpi=180,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )
    plt.close(fig)
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmark_results/pp_reconfigure/acceptance/llama7b-v2.log"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/pp_reconfigure/plots/reconfiguration-timeline"),
    )
    args = parser.parse_args()
    plot(args.report, args.output)
