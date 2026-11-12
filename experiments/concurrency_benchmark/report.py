"""Aggregate complete repeats, selecting a single whole-run static boundary."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def main():
    results = []
    for path in sorted((ROOT / "results").glob("*/summary.json")):
        r = json.loads(path.read_text())
        events = [e for p in (path.parent / "decisions").glob("*.log") for e in rows(p)]
        migrations = [e for e in events if e.get("status") == "resumed"]
        r["migrations"] = len(migrations)
        r["pause_ms"] = sum(e["result"]["pause_ms"] for e in migrations)
        results.append(r)
        samples = rows(path.parent / "samples.jsonl")
        if r["label"] in ("full", "threshold"):
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            t = [s["time"] - r["start"] for s in samples]
            axes[0].step(
                t, [s["workers"][0]["layer_range"][1] for s in samples], where="post"
            )
            axes[0].set_ylabel("PP boundary")
            for key in ("running_requests", "waiting_requests", "client_inflight"):
                axes[1].plot(t, [s[key] for s in samples], label=key)
            axes[1].legend()
            axes[2].plot(t, [s["kv_cache_usage"] for s in samples], label="KV usage")
            axes[2].legend()
            axes[2].set_xlabel("Seconds")
            for ax in axes:
                for x in (200, 400, 600):
                    ax.axvline(x, color="gray", linestyle=":")
            fig.tight_layout()
            fig.savefig(path.parent / "timeline.png")
            plt.close(fig)
    if not results:
        raise RuntimeError("No completed trials")
    aggregates = []
    for c in sorted({r["concurrency"] for r in results}):
        for label in sorted({r["label"] for r in results}):
            group = [
                r for r in results if r["concurrency"] == c and r["label"] == label
            ]
            if not group:
                continue
            a = dict(concurrency=c, label=label, repeats=len(group))
            for key in (
                "throughput",
                "goodput",
                "tpot_ms_p50",
                "tpot_ms_p99",
                "ttft_ms_p50",
                "ttft_ms_p99",
                "migrations",
                "pause_ms",
                "preemptions",
                "drain_seconds",
            ):
                a[key] = float(np.mean([r[key] for r in group]))
                a[key + "_std"] = (
                    float(np.std([r[key] for r in group], ddof=1))
                    if len(group) > 1
                    else 0
                )
            aggregates.append(a)
    best = {}
    for c in sorted({a["concurrency"] for a in aggregates}):
        static = [
            a
            for a in aggregates
            if a["concurrency"] == c and a["label"].startswith("static_")
        ]
        if static:
            best[c] = max(static, key=lambda a: a["throughput"])

    def series(label):
        return sorted(
            (
                [dict(v, label="best_static") for v in best.values()]
                if label == "best_static"
                else [a for a in aggregates if a["label"] == label]
            ),
            key=lambda a: a["concurrency"],
        )

    for key in ("throughput", "goodput", "tpot_ms_p50", "tpot_ms_p99"):
        fig, ax = plt.subplots()
        for label in ("full", "threshold", "best_static"):
            group = series(label)
            ax.errorbar(
                [a["concurrency"] for a in group],
                [a[key] for a in group],
                yerr=[a[key + "_std"] for a in group],
                marker="o",
                label=label,
            )
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(key)
        ax.legend()
        fig.tight_layout()
        fig.savefig(ROOT / f"{key}.png")
        plt.close(fig)
    fig, ax = plt.subplots()
    full = [a for a in series("full") if a["concurrency"] in best]
    for key in ("throughput", "goodput"):
        ax.plot(
            [a["concurrency"] for a in full],
            [
                a[key] / best[a["concurrency"]][key]
                if best[a["concurrency"]][key]
                else float("nan")
                for a in full
            ],
            marker="o",
            label=key,
        )
    ax.axhline(1, color="gray")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Full / Best Static")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "speedup.png")
    plt.close(fig)
    (ROOT / "aggregate.json").write_text(
        json.dumps(dict(aggregates=aggregates, best_static=best), indent=2)
    )
    text = [
        "# 动态 PP 并发扫描",
        f"已完成 {len(results)}/243 组。均值与样本标准差来自独立重复。",
        "固定窗口为 600 秒；"
        "只计窗口内收到的输出 Token。"
        "Goodput 对完整请求按 TTFT≤5000ms、TPOT≤100ms 分类，再计其窗口内 Token。"
        "Drain 不进入分母，跨窗口请求延迟使用完整响应；"
        "TPOT 分位数是逐请求平均 TPOT 的分位数。"
        "",
        "Best Static 按每个并发下全程吞吐的重复均值选择唯一切分点，"
        "Goodput 与 Speedup 使用同一切分点。"
        "aggregate.json 保留全部切分点，支持另行分析 Goodput 最优点。"
        "",
        "所有组输入32、输出96 Token，闭环持续补充；"
        "保留原有128块上限、批Token上限128，max-num-seqs=64。"
        "高并发排队及抢占是实际容量限制的测量结果。"
        "模拟延迟不等同GPU降频。"
        "",
        "|并发|策略|重复|Token/s|Goodput|TPOT P50/P99 ms|"
        "TTFT P50/P99 ms|迁移|Pause ms|抢占|",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(best):
        for a in [
            a
            for a in aggregates
            if a["concurrency"] == c and a["label"] in ("full", "threshold")
        ] + [best[c]]:
            text.append(
                f"|"
                f"{c}|"
                f"{a['label']}|"
                f"{a['repeats']}|"
                f"{a['throughput']:.2f}|"
                f"{a['goodput']:.2f}|"
                f"{a['tpot_ms_p50']:.1f}/{a['tpot_ms_p99']:.1f}|"
                f"{a['ttft_ms_p50']:.1f}/{a['ttft_ms_p99']:.1f}|"
                f"{a['migrations']:.1f}|"
                f"{a['pause_ms']:.1f}|"
                f"{a['preemptions']:.1f}|"
            )
    for label in ("full", "threshold", "best_static"):
        group = series(label)
        if group:
            peak = max(group, key=lambda a: a["throughput"])
            saturation = min(
                a["concurrency"]
                for a in group
                if a["throughput"] >= 0.95 * peak["throughput"]
            )
            text.append(
                f"{label} 实测峰值 {peak['throughput']:.2f} Token/s"
                f"（并发 {peak['concurrency']}）；"
                f"首次达到峰值95%的并发为 {saturation}，作为扫描范围内饱和点。"
            )
    for a in full:
        b = best[a["concurrency"]]
        text.append(
            f"并发 {a['concurrency']}：Full / {b['label']} 吞吐比 "
            f"{a['throughput'] / b['throughput']:.3f}；"
            f"平均迁移 {a['migrations']:.1f} 次，Pause {a['pause_ms']:.1f}ms。"
            f"收益包含实际迁移与排队代价；"
            f"低于1即未超过最佳固定PP。"
        )
    text += [
        f"![{key}]({key}.png)"
        for key in ("throughput", "goodput", "tpot_ms_p50", "tpot_ms_p99", "speedup")
    ]
    text.append(
        "逐组分层时间线位于 results/*/timeline.png；"
        "samples.jsonl 包含 Running/Waiting、GPU利用率、KV使用率，"
        "phases.jsonl 包含计划和实际切换时间。"
        "所有原始请求、模型更新、迁移校验、服务日志均保留。"
    )
    (ROOT / "benchmark_report.md").write_text("\n\n".join(text) + "\n")


if __name__ == "__main__":
    main()
