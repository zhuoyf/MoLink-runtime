# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive results from raw records; never substitute simulated service metrics."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from vllm.distributed.pp_reconfigure.adaptive.model import PerformanceModel


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def mean(values):
    return statistics.mean(values) if values else None


parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()
out = Path(args.out)
profile = json.loads((out / "profile.json").read_text())
model_name = Path(profile["conditions"]["model"]).name
metrics = {}
groups = defaultdict(list)
for row in rows(out / "summary.jsonl"):
    label = row["label"]
    if label.startswith("comparison_"):
        groups[label.split("_", 2)[2]].append(row)
    outcomes = rows(out / label / "outcome.log")
    migrations = [r["result"] for r in outcomes if r["status"] == "resumed"]
    samples = rows(out / (label + "_samples.jsonl"))
    requests = rows(out / (label + "_requests.jsonl"))
    preemptions, pressure = [], []
    for sample in samples:
        for worker in sample.get("workers", []):
            m = worker["metrics"]
            preemptions.append(m.get("preemptions", 0))
            pressure.append(m.get("kv_cache_usage", 0))
    metrics[label] = {
        **row,
        "migrations": len(migrations),
        "migration_total_ms": sum(r["total_ms"] for r in migrations),
        "migration_pause_ms": sum(r["pause_ms"] for r in migrations),
        "preemptions_delta": max(preemptions, default=0) - min(preemptions, default=0),
        "max_kv_pressure": max(pressure, default=0),
        "request_count": len(requests),
        "all_requests_complete": all(len(r["token_ids"]) == 256 for r in requests),
        "errors": rows(out / (label + "_errors.jsonl")),
    }
# Prequential errors: predict each distinct sample before its EWMA update.
model = PerformanceModel()
seen = set()
errors = defaultdict(list)
for report in rows(out / "profile_samples.jsonl"):
    for worker in report.get("workers", []):
        w = worker["metrics"]
        for sample in w.get("workload_samples", []):
            if sample["timestamp"] < report.get("model_sample_not_before", 0):
                continue
            identity = (w["stage"], sample["sequence"])
            if identity in seen:
                continue
            seen.add(identity)
            key = model.compute_key(
                w["stage"],
                sample["layers"],
                sample["phase"],
                sample["batch"],
                sample["context"],
                sample["tokens"],
            )
            predicted = model.compute(
                w["stage"],
                sample["layers"],
                sample["phase"],
                sample["batch"],
                sample["context"],
                sample["tokens"],
            )
            observed = sample["gpu_ms"]
            if predicted is not None and observed > 0:
                errors[sample["phase"]].append(abs(predicted - observed) / observed)
            model.update(key, observed, sample["timestamp"])
accuracy = {
    k: {
        "count": len(v),
        "mean_absolute_relative_error": mean(v),
        "p95_relative_error": sorted(v)[int(0.95 * (len(v) - 1))],
    }
    for k, v in errors.items()
}
# Same prompt, deterministic seed, and batch-invariant execution: compare IDs.
reference = {}
mismatches = []
for path in sorted(out.glob("comparison_*_requests.jsonl")):
    for row in rows(path):
        key = (row["context"], len(row["token_ids"]))
        if key in reference and reference[key] != row["token_ids"]:
            mismatches.append(
                {"file": path.name, "index": row["index"], "context": row["context"]}
            )
        reference.setdefault(key, row["token_ids"])
convergence = defaultdict(list)
for update in rows(out / "online_update" / "model_update.log"):
    for key, estimate in update.get("estimates", {}).items():
        if key.startswith("compute/"):
            convergence[key].append(estimate)
convergence_summary = {
    key: {"updates": len(values), "initial": values[0], "final": values[-1]}
    for key, values in convergence.items()
    if len(values) > 1
}
summary = {
    "online_convergence": convergence_summary,
    "runs": metrics,
    "prediction_error": accuracy,
    "token_mismatches": mismatches,
}
(out / "analysis.json").write_text(json.dumps(summary, indent=2))
lines = [
    "# 动态 PP 调度器实验报告",
    "",
    "数据来源：双 GPU 实际 HTTP 推理与 NCCL 测量；原始记录位于本目录。",
    "",
    "## 对照结果",
    "",
    "每个策略使用相同开放环到达序列：0/3/6/9 秒分别提交 1/4/8/1 个请求，"
    "上下文 32/128/64/32 Token，"
    "各生成 256 Token。"
    "固定 seed=42、temperature=0，"
    "每组重置前缀缓存并预热；多轮运行时反转执行顺序。"
    "",
    "",
    "|策略|重复数|吞吐 Token/s 均值|TPOT ms 均值|SLO Goodput Token/s|迁移次数合计|",
    "|---|---:|---:|---:|---:|---:|",
]
if groups and all(len(values) == 1 for values in groups.values()):
    lines[4:4] = ["本次每个策略仅运行一轮，结果不提供统计显著性或跨负载收益保证。", ""]
for label, values in sorted(groups.items()):
    lines.append(
        f"|{label}|{len(values)}|"
        f"{mean([v['throughput_tokens_s'] for v in values]):.2f}|"
        f"{mean([v['mean_tpot_ms'] for v in values]):.2f}|"
        f"{mean([v['slo_goodput_tokens_s'] for v in values]):.2f}|"
        f"{sum(metrics[v['label']]['migrations'] for v in values)}|"
    )
static = {
    k: mean([v["throughput_tokens_s"] for v in vals])
    for k, vals in groups.items()
    if k.startswith("static_")
}
if static:
    lines += [
        "",
        "Best Static（本测试所有合法固定边界的事后最优值）为 "
        f"{max(static, key=static.get)}。"
        "SLO 定义为 TTFT≤5000ms 且 TPOT≤150ms。"
        "",
    ]
lines += [
    "",
    "## 资源与迁移成本",
    "",
    "|策略|抢占增量合计|KV 压力峰值|迁移总耗时 ms|暂停总耗时 ms|",
    "|---|---:|---:|---:|---:|",
]
for label, values in sorted(groups.items()):
    measured = [metrics[v["label"]] for v in values]
    lines.append(
        f"|{label}|{sum(v['preemptions_delta'] for v in measured)}|"
        f"{max(v['max_kv_pressure'] for v in measured):.3f}|"
        f"{sum(v['migration_total_ms'] for v in measured):.2f}|"
        f"{sum(v['migration_pause_ms'] for v in measured):.2f}|"
    )
lines += [
    "",
    "## 在线更新与自动迁移",
    "",
    f"稳定尾段有 {len(convergence_summary)} 个 Compute 桶获得多次更新；"
    "初值、终值、样本数和误差保存在 analysis.json 的 online_convergence。",
]
if "scheduler" in metrics:
    active = metrics["scheduler"]
    lines += [
        "",
        f"独立性能变化实验：{active['migrations']} 次自动迁移，"
        f"事务合计 {active['migration_total_ms']:.2f}ms，"
        f"暂停合计 {active['migration_pause_ms']:.2f}ms。",
    ]

lines += [
    "",
    "## 预测和正确性",
    "",
    f"逐样本先预测后更新的误差：`{json.dumps(accuracy, ensure_ascii=False)}`。"
    "首个无模型样本不计入误差。"
    "这是在线预序误差，"
    "不是独立数据集泛化精度。"
    "",
    "",
    f"相同上下文的逐 Token 不一致记录：{len(mismatches)}。"
    "完整请求计数、抢占增量、KV 峰值、迁移总耗时和暂停耗时见 `analysis.json`。"
    "",
    "",
    "## 解释与边界",
    "",
    f"当前使用两张 RTX 3090 Ti 与 {model_name} 校验端到端行为。"
    "历史 Llama-7B 迁移数据保存为独立 historical_profile.json，"
    "按模型、dtype 与环境条件隔离，不直接混入当前计算分桶。"
    "",
    "",
    "模型使用固定开销＋每层开销曲线，并用分桶 EWMA 更新速度倍率。"
    "CUDA 事件从 V2 的 PP 接收完成后开始计时，"
    "包含模型执行与主机发射间隙，"
    "不能当作纯算子时间。"
    "小消息网络观测包含主机 staging，"
    "迁移通信统计提供大消息有效带宽。"
    "冷启动未覆盖的工作负载保持当前分层。"
    "空载迁移的干扰值采用 1.0 保守上界，"
    "在线迁移后以实际执行步数更新。"
    "",
    "",
    "security_mode=off。可信执行、加密、证明验证器均未实现；可信策略必须提供有效节点能力且需要实际安全成本提供器才能参与决策。",
    "",
]
(out / "scheduler_report.md").write_text("\n".join(lines))
print(out / "scheduler_report.md")
