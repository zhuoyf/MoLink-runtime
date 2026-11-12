# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate measured comparisons and timelines, including incomplete/failed runs."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def rows(path):
    return (
        [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    )


def main():
    config = json.loads((ROOT / "frozen.json").read_text())
    best = json.loads((ROOT / "pilot/phase_best.json").read_text())
    groups = defaultdict(list)
    analysis = {}
    reference = json.loads(
        next((ROOT / "pilot/profile").glob("A_16_requests.jsonl"))
        .read_text()
        .splitlines()[0]
    )["token_ids"]
    timelines = []
    for directory in sorted((ROOT / "results").glob("*")):
        if not (directory / "summary.json").exists():
            continue
        summary = json.loads((directory / "summary.json").read_text())
        requests = rows(directory / "requests.jsonl")
        samples = rows(directory / "samples.jsonl")
        decisions = rows(directory / "decisions/decision.log")
        outcomes = [
            x
            for x in rows(directory / "decisions/outcome.log")
            if x["status"] == "resumed"
        ]
        outcomes += [
            x for x in rows(directory / "oracle_migrations.jsonl") if x.get("result")
        ]
        summary["migrations"] = len(outcomes)
        summary["migration_ms"] = sum(x["result"]["total_ms"] for x in outcomes)
        summary["pause_ms"] = sum(x["result"]["pause_ms"] for x in outcomes)
        summary["token_mismatches"] = sum(x["token_ids"] != reference for x in requests)
        summary["ttft_slo_failures"] = sum(
            r["ttft_ms"] > config["slo"]["ttft_ms"] for r in requests
        )
        summary["tpot_slo_failures"] = sum(
            r["tpot_ms"] > config["slo"]["tpot_ms"] for r in requests
        )
        summary["max_kv_pressure"] = max(
            (w["kv_cache_usage"] for x in samples for w in x["workers"]), default=0
        )
        summary["queue_peak"] = max(
            (
                x.get("waiting_requests", max(0, len(x.get("requests", {})) - 4))
                for x in samples
            ),
            default=0,
        )
        summary["extra_prefill_tokens"] = max(
            x["prefill_tokens"] for x in summary["counters"]
        ) - sum(x["context"] for x in requests)
        summary["resumed_requests"] = max(
            x["resumed_requests"] for x in summary["counters"]
        )
        phases = rows(directory / "phases.jsonl")
        from collections import Counter

        summary["decision_reasons"] = dict(Counter(x.get("reason") for x in decisions))
        summary["phases"] = {}
        for i, phase in enumerate(phases):
            begin = phase["planned"]
            end = begin + config["phase_seconds"]
            selected = [
                x for x in decisions if begin <= x["time"] < end and x.get("candidates")
            ]
            predictions = []
            for d in selected:
                valid = [
                    c for c in d["candidates"] if c.get("valid") and "cost_seconds" in c
                ]
                if valid:
                    predictions.append(
                        min(valid, key=lambda c: c["cost_seconds"])["boundary"]
                    )
            hits = [
                x
                for x in samples
                if begin <= x["time"] < end
                and x["workers"][0]["layer_range"][1] == best[phase["phase"]]
            ]
            deliveries = [t for r in requests for t in r["times"] if begin <= t < end]
            good_deliveries = [
                t
                for r in requests
                if r["ttft_ms"] <= config["slo"]["ttft_ms"]
                and r["tpot_ms"] <= config["slo"]["tpot_ms"]
                for t in r["times"]
                if begin <= t < end
            ]
            summary["phases"][phase["phase"]] = {
                "measured_best": best[phase["phase"]],
                "predicted_best_sequence": predictions,
                "response_seconds": min(
                    (x["time"] - begin for x in hits), default=None
                ),
                "phase_application_lag": phase["time"] - begin,
                "delivered_tokens": len(deliveries),
                "good_tokens": len(good_deliveries),
            }
        for x in samples:
            timelines.append(
                {
                    "trial": directory.name,
                    "seconds": x["time"] - summary["start"],
                    "boundary": x["workers"][0]["layer_range"][1],
                    "kv_pressure": max(w["kv_cache_usage"] for w in x["workers"]),
                }
            )
        analysis[directory.name] = summary
        groups[summary["label"]].append(summary)
    (ROOT / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (ROOT / "timeline.json").write_text(json.dumps(timelines, indent=2))
    lines = [
        "# 动态异构 GPU 调度实验",
        "",
        "正式实验参数及请求轨迹冻结于 frozen.json；所有策略相同"
        "到达序列、模型、SLO、缓存设置及环境。",
        "",
        "## 阶段与对照",
        "",
        f"独立 Profiling 的阶段最佳边界：{best}。阶段 B "
        f"为 GPU1 降速/限 KV，阶段 C 对调。",
        "每阶段 120 秒；每 12 秒到达 4 个请求，各 32 Pro"
        "mpt / 96 Output Token。SLO：TTFT≤50"
        "00ms，TPOT≤100ms。",
        "完整策略与阈值、先知策略从 16/16 启动；固定策略分别从自身边"
        "界启动。full_start28 单独量化启动配置影响。",
        "Oracle 为已知阶段最佳边界的实测策略，计入真实迁移；另提供忽"
        "略迁移和响应时间的经验理想上界，不能视为严格数学上界。",
        "",
        "## 全程实测",
        "",
        "|策略|完成重复数|SLO Goodput Token/s 均值±"
        "标准差|吞吐|TPOT ms|TTFT ms|请求延迟 ms|迁移"
        "数均值|暂停 ms 均值|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ideal = {}
    for phase, boundary in best.items():
        measured = json.loads(
            (ROOT / f"pilot/profile/{phase}_{boundary}.json").read_text()
        )
        ideal[phase] = min(32.0, measured["goodput"])
    lines[11:11] = ["",
        f"忽略响应、迁移与跨阶段排队的经验理想 Goodput 上界：{ideal}"
        f" Token/s，阶段均值 {statistics.mean(ideal.values()):.2f}"
        f"。"
    ]
    means = {}
    for label, runs in sorted(groups.items()):
        avg = lambda key, runs=runs: statistics.mean(x[key] for x in runs)
        means[label] = avg("goodput")
        sd = statistics.stdev(x["goodput"] for x in runs) if len(runs) > 1 else 0
        lines.append(
            f"|{label}|{len(runs)}|{avg('goodput'):.2f}"
            f" ± {sd:.2f}|{avg('throughput'):.2f}"
            f"|{avg('tpot_ms'):.2f}|{avg('ttft_ms'):.2f}"
            f"|{avg('latency_ms'):.2f}|{avg('migrations'):.2f}"
            f"|{avg('pause_ms'):.2f}|"
        )
    complete = len(groups) == 14 and all(len(x) == 5 for x in groups.values())
    completion = (
        "所有 14 组各 5 次完成" if complete else "尚未全部完成，不能作为最终结论"
    )
    lines += [
        "",
        f"完成性：{completion}。",
    ]
    static = {k: v for k, v in means.items() if k.startswith("static_")}
    if static and "full" in means:
        winner = max(static, key=static.get)
        delta = means["full"] - static[winner]
        lines += [
            f"全程 Best Static={winner}，完整策略 Goodput 差值 {delta:+.2f} Token/s。",
            "Best Static 按全部阶段的整体 Goodput 选取，不能拼接各阶段不同的固定策略。",
        ]
        for baseline, interpretation in [
            ("without_compute", "计算性能感知"),
            ("compute_only", "KV 风险项"),
            ("without_migration_cost", "迁移成本项"),
            ("full_start28", "初始切分位置"),
        ]:
            if baseline in means:
                lines.append(
                    f"{interpretation}：full 相对 {baseline}"
                    f" 的 Goodput 差值 {means['full'] - means[baseline]:+.2f}"
                    f" Token/s；零或负收益同样保留。"
                )
        full_runs = groups["full"]
        lines += ["", "阶段诊断（完整策略）："]
        for phase in "ABC":
            responses = [r["phases"][phase]["response_seconds"] for r in full_runs]
            observed = [r for r in responses if r is not None]
            tokens = statistics.mean(
                r["phases"][phase]["good_tokens"] for r in full_runs
            )
            delay = statistics.mean(observed) if observed else None
            lines.append(
                f"{phase}：平均成功 Token={tokens:.1f}，"
                f"首次到达实测最佳的平均秒数={delay}，"
                f"未到达次数={responses.count(None)}。"
            )
        ttft = sum(r["ttft_slo_failures"] for r in full_runs)
        tpot = sum(r["tpot_slo_failures"] for r in full_runs)
        migration = statistics.mean(r["migration_ms"] for r in full_runs)
        lines.append(
            f"完整策略 TTFT 超标 {ttft} 次，TPOT 超标 {tpot} 次；"
            f"每轮迁移总耗时均值 {migration:.1f}ms。"
        )
        if delta <= 0:
            lines.append(
                "完整策略本次未超过 Best Static。"
                "上述阶段未到达/响应时间及 SLO 超标计数是直接诊断证据；"
                "各消融差值用于区分预测项贡献，不能把全部差距归于迁移暂停。"
            )
        # Paired cumulative successful tokens: empirical break-even vs best static.
        break_even = {}
        for repeat in range(5):
            full_dir = ROOT / f"results/{repeat}_full"
            static_dir = ROOT / f"results/{repeat}_{winner}"
            if (
                not (full_dir / "summary.json").exists()
                or not (static_dir / "summary.json").exists()
            ):
                continue
            series = []
            for d in (full_dir, static_dir):
                start = json.loads((d / "summary.json").read_text())["start"]
                series.append(
                    sorted(
                        t - start
                        for r in rows(d / "requests.jsonl")
                        if r["ttft_ms"] <= 5000 and r["tpot_ms"] <= 100
                        for t in r["times"]
                    )
                )
            import bisect

            horizon = int(max([360] + [s[-1] for s in series if s]))
            gains = [
                bisect.bisect_right(series[0], t) - bisect.bisect_right(series[1], t)
                for t in range(horizon + 1)
            ]
            last_loss = max((i for i, g in enumerate(gains) if g <= 0), default=-1)
            break_even[str(repeat)] = last_loss + 1 if last_loss < horizon else None
        (ROOT / "break_even.json").write_text(json.dumps(break_even, indent=2))
        lines += [
            "",
            f"相对全程 Best Static 的累计 SLO 成功 Token"
            f" 永久转正时间（含实际迁移损失；null 表示未回本）：{break_even}"
            f"。",
            "此经验回本时间不是迁移单独因果效应；逐迁移预测成本、实际耗时及暂停"
            "保留在原始决策日志。",
        ]
    lines += [
        "",
        "## 正确性与解释",
        "",
        f"Token 不一致合计：{sum(x['token_mismatches'] for x in analysis.values())}"
        f"；抢占合计：{sum(x['preemptions'] for x in analysis.values())}"
        "；额外 Prefill Token："
        f"{sum(x['extra_prefill_tokens'] for x in analysis.values())}"
        f"。",
        "analysis.json 包含各阶段预测最优序列、实测最优、响应"
        "时间、KV 峰值、排队、抢占和原始计数。",
        "若完整策略未胜出，应结合 EWMA 响应延迟、未迁移原因、迁移次数"
        "和暂停、阶段耗时以及到达率上限解释；不能仅凭一次均值归因。",
        "消融仍执行相同物理容量和安全可行性检查；compute_only "
        "仅去掉 KV 风险代价，without_compute 去掉候选计"
        "算差异，without_migration_cost 去掉预测迁移"
        "代价。",
        "模型/延迟模拟代表受控假设环境，不外推为真实异型号 GPU 的通用收益。",
        "",
        "![性能对比](comparison.png)",
        "",
        "![分层时间线](timeline.png)",
    ]
    if means:
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(means.keys(), means.values())
        ax.set_ylabel("SLO Goodput (tokens/s)")
        ax.tick_params(axis="x", rotation=65)
        fig.tight_layout()
        fig.savefig(ROOT / "comparison.png")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        for phase in "ABC":
            values = [
                json.loads((ROOT / f"pilot/profile/{phase}_{b}.json").read_text())
                for b in range(4, 32, 4)
            ]
            ax.plot(
                [x["boundary"] for x in values],
                [x["goodput"] for x in values],
                marker="o",
                label=phase,
            )
        ax.set(xlabel="Stage 0 layers", ylabel="Pilot SLO Goodput (tokens/s)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(ROOT / "comparison.png")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    for label in ("full", "threshold", "oracle", "full_start28"):
        values = [x for x in timelines if x["trial"] == f"0_{label}"]
        if values:
            ax.step(
                [x["seconds"] for x in values],
                [x["boundary"] for x in values],
                where="post",
                label=label,
            )
    if not timelines and (ROOT / "pilot/dynamic/summary.json").exists():
        pilot = json.loads((ROOT / "pilot/dynamic/summary.json").read_text())
        values = rows(ROOT / "pilot/dynamic/samples.jsonl")
        ax.step(
            [x["time"] - pilot["start"] for x in values],
            [x["workers"][0]["layer_range"][1] for x in values],
            where="post",
            label="Pilot full (60s phases)",
        )
        ax.axvline(60, color="gray", linestyle=":")
        ax.axvline(120, color="gray", linestyle=":")
    for t in (120, 240) if timelines else ():
        ax.axvline(t, color="gray", linestyle="--")
    ax.set(xlabel="Seconds", ylabel="Stage 0 layers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "timeline.png")
    plt.close(fig)
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")
    if complete:
        assert all(
            x["complete"]
            and x["token_mismatches"] == 0
            and x["extra_prefill_tokens"] == 0
            and x["preemptions"] == 0
            for x in analysis.values()
        )


if __name__ == "__main__":
    main()
