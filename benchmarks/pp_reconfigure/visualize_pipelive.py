# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot measured token delivery, live resizing, patching and cutover costs."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def records(row, phase):
    return [
        json.loads(line)
        for line in (Path(row["log_dir"]) / f"{phase}.log").read_text().splitlines()
    ]


def phase_time(row, phase, event):
    return next(
        item["time"] for item in records(row, phase) if item.get("event") == event
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison")
    parser.add_argument("--target", type=int, default=18)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.comparison).read_text())
    checked = json.loads(Path(args.verification).read_text())
    assert data["status"] == checked["status"] == "passed"
    measured = data["results"][4 * args.warmup_iterations :]
    rows = [r for r in measured if r["boundary"] == args.target]
    summary = {}
    for mode in ("sync", "async"):
        trials = [r for r in rows if r["mode"] == mode]
        assert trials
        summary[mode] = {"runs": len(trials)}
        for key in ("total_ms", "pause_ms"):
            summary[mode][key] = statistics.mean(r[key] for r in trials)
        summary[mode]["communication_ms"] = statistics.mean(
            r["metrics"]["communication_ms"] for r in trials
        )
        summary[mode]["forward_span_overlap_fraction"] = statistics.mean(
            r["metrics"]["overlap_fraction"] for r in trials
        )
        for row in trials:
            tokens = json.loads((Path(row["log_dir"]) / "tokens.json").read_text())
            assert tokens["actual"] == tokens["expected"]
            assert all(len(t) == 192 for t in tokens["actual"])
            assert all(len(t) == 192 for t in row["stream_token_times"])
    for row in checked["results"]:
        digests = [w["kv_digest"] for w in row["kv_verification"]]
        assert len(digests) == 2 and len(set(digests)) == 1

    # The first measured pair is shown; bars below use every forward trial.
    sync = next(r for r in rows if r["mode"] == "sync")
    live = next(r for r in rows if r["mode"] == "async")
    patches = records(live, "patch")
    incremental = [p for p in patches if not p["full"] and p["bytes"] > 0]
    assert incremental and sum(p["bytes"] for p in incremental) > 0
    start = phase_time(live, "start_live", "begin")
    pause = phase_time(live, "pause", "begin")
    resume = phase_time(live, "resume", "end")
    delivered = [
        sum(start <= t < pause for t in ts) for ts in live["stream_token_times"]
    ]
    assert all(n > 0 for n in delivered), delivered
    summary["async_tokens_delivered_before_pause"] = delivered
    summary["incremental_patches"] = len(incremental)
    summary["incremental_bytes"] = sum(p["bytes"] for p in incremental)
    rounds = [p for p in incremental if p["last"]]
    summary["patch_rounds_before_pause_request"] = sum(
        p["time"] < pause for p in rounds
    )
    pause_end = phase_time(live, "pause", "end")
    summary["patch_rounds_during_drain"] = sum(
        pause <= p["time"] < pause_end for p in rounds
    )
    summary["pause_reduction"] = (
        1 - summary["async"]["pause_ms"] / summary["sync"]["pause_ms"]
    )

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    ax = axes[0, 0]
    ends = []
    for offset, row, color in ((0, sync, "#64748b"), (4, live, "#0891b2")):
        origin = phase_time(row, "validate", "begin")
        stop = phase_time(row, "pause", "begin")
        go = phase_time(row, "resume", "end")
        ends.append((go - origin) * 1000)
        ax.broken_barh(
            [((stop - origin) * 1000, (go - stop) * 1000)],
            (offset - 0.4, 2.8),
            facecolors="#fecaca",
            label="Keep pause + drain" if offset == 0 else None,
        )
        if row["mode"] == "async":
            ax.broken_barh(
                [((start - origin) * 1000, (pause - start) * 1000)],
                (offset - 0.4, 2.8),
                facecolors="#cffafe",
                label="Background migration",
            )
        for i, ts in enumerate(row["stream_token_times"]):
            xs = [(t - origin) * 1000 for t in ts]
            ax.plot(xs, [offset + i] * len(xs), "|", color=color, markersize=9)
    ax.set_yticks(
        [0, 1, 2, 4, 5, 6],
        ["Sync R1", "Sync R2", "Sync R3", "Async R1", "Async R2", "Async R3"],
    )
    ax.set_xlim(-200, max(ends) + 350)
    ax.set_xlabel("Time from migration request (ms); each tick = one delivered token")
    ax.set_title("A. Do requests keep producing tokens during migration?")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[0, 1]
    for offset, mode, color in ((-0.18, "sync", "#64748b"), (0.18, "async", "#0891b2")):
        values = [
            summary[mode][k] for k in ("total_ms", "pause_ms", "communication_ms")
        ]
        bars = ax.bar(
            [x + offset for x in range(3)], values, width=0.34, color=color, label=mode
        )
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_xticks(range(3), ["Total migration", "Inference pause", "NCCL send events"])
    ax.set_ylabel("Mean time (ms)")
    ax.set_title(f"B. Pause reduction: {summary['pause_reduction']:.1%}")
    ax.margins(y=0.18)
    ax.legend()

    ax = axes[1, 0]
    states = next(r["resize_states"] for r in data["results"] if r["resize_states"])
    assert [s["num_blocks"] for s in states] == [64, 48, 64]
    for before, after in zip(states, states[1:]):
        assert before["requests"].keys() == after["requests"].keys()
        for request_id, request in before["requests"].items():
            current = after["requests"][request_id]
            assert current["num_preemptions"] == 0
            assert current["num_computed_tokens"] >= request["num_computed_tokens"]
            for old, new in zip(request["block_ids"], current["block_ids"]):
                assert old == new[: len(old)], "resize changed live block mappings"
    for stage, color in ((0, "#7c3aed"), (1, "#0891b2")):
        sizes = [
            next(w for w in s["workers"] if w["stage"] == stage)["kv_capacity_bytes"]
            / (1 << 20)
            for s in states
        ]
        assert sizes[1] < sizes[0] and sizes[2] == sizes[0]
        ax.plot(range(3), sizes, "o-", color=color, label=f"GPU {stage} allocated KV")
        for x, y in enumerate(sizes):
            ax.annotate(
                f"{y:.0f} MiB",
                (x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
            )
    ax.set_xticks(
        range(3),
        ["64 blocks\nbefore", "48 blocks\nafter shrink", "64 blocks\nafter expansion"],
    )
    ax.set_ylabel("Physical stacked KV allocations (MiB)")
    ax.set_title("C. Online resize with 3 active requests, zero preemptions")
    ax.margins(y=0.22)
    ax.legend()

    ax = axes[1, 1]
    progress = records(live, "progress")
    scheduled, applied, xs = [0], [0], [0.0]
    for p in progress:
        source = next(w for w in p["workers"] if w["stage"] == w["source"])
        dest = next(w for w in p["workers"] if w["stage"] != w["source"])
        xs.append((p["time"] - start) * 1000)
        scheduled.append(source["scheduled"])
        applied.append(dest["applied"])
    final = live["load_and_transfer"]
    src = next(w for w in final if w["stage"] == w["source"])
    dst = next(w for w in final if w["stage"] != w["source"])
    assert src["scheduled"] == dst["applied"]
    xs.append((phase_time(live, "sync_live", "end") - start) * 1000)
    scheduled.append(src["scheduled"])
    applied.append(dst["applied"])
    ax.step(
        xs,
        scheduled,
        where="post",
        label="Polled source token watermark",
        color="#64748b",
    )
    ax.step(
        xs,
        applied,
        where="post",
        label="Polled destination applied watermark",
        color="#0891b2",
    )
    last = [p for p in incremental if p["last"]]
    ax.scatter(
        [(p["time"] - start) * 1000 for p in last],
        [p["watermark"] for p in last],
        marker="x",
        color="#7c3aed",
        label="Incremental round applied",
        zorder=5,
    )
    full_end = max(p["time"] for p in patches if p["full"])
    ax.axvspan(
        0,
        (full_end - start) * 1000,
        color="#e2e8f0",
        alpha=0.5,
        label="Historical KV transfer",
    )
    ax.axvspan(
        (pause - start) * 1000, (resume - start) * 1000, color="#fecaca", alpha=0.6
    )
    ax.set_xlabel("Time from background migration start (ms)")
    ax.set_ylabel("Polled migration-relative tokens (all 3 requests)")
    ax.set_title(f"D. Incremental patches catch up; final lag = 0 ({len(last)} rounds)")
    ax.legend(fontsize=8)
    fig.suptitle(
        "PipeLive live KV migration | Llama-2-7B FP16 | 2 x RTX 3090 Ti\n"
        "PP 20/12 -> 18/14 | 3 requests x 192 exact tokens",
        fontsize=15,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output / f"experiment.{extension}", dpi=170)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# PipeLive 可视化验证实验",
        "",
        "配置：Llama-2-7B FP16，双 RTX 3090 Ti，PP 20/12 → 18/14，"
        "迁移第 18–19 层；三个活跃请求，每个生成 192 tokens。",
        "",
        "第一轮完整往返作为预热；下表来自后续正向迁移，哈希校验单独运行。",
        "",
        "| 模式 | 次数 | 总耗时 ms | 暂停 ms | NCCL ms | Forward 窗口重叠 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("sync", "async"):
        s = summary[mode]
        lines.append(
            f"| {mode} | {s['runs']} | {s['total_ms']:.2f} | "
            f"{s['pause_ms']:.2f} | {s['communication_ms']:.2f} | "
            f"{s['forward_span_overlap_fraction']:.1%} |"
        )
    lines += [
        "",
        f"暂停降低 {summary['pause_reduction']:.1%}。",
        "",
        "![实测时间线与 KV 状态](experiment.png)",
        "",
        "读图：A 中每条刻度表示客户端收到一个 token，蓝色区域为后台迁移，"
        "红色区域为 keep pause 到 resume；B 为预热后的耗时均值；"
        "C 为活跃请求期间 64→48→64 blocks 的实际 KV 分配量；"
        "D 为源端写入与目标应用的 token 水位，以及完成应用的增量补丁轮次。",
        "",
        f"图示异步试验中，三个请求在后台迁移开始后、最终暂停前分别收到 "
        f"{delivered} tokens；发送 {len(incremental)} 个非空增量分层补丁，"
        f"共 {summary['incremental_bytes'] / (1 << 20):.3f} MiB；最终水位差为零。",
        "",
        f"该示例在最终暂停请求前完成的非空增量轮次为 "
        f"{summary['patch_rounds_before_pause_request']}，"
        f"在排空阶段完成的轮次为 {summary['patch_rounds_during_drain']}。"
        "这些数据验证增量补丁与最终一致性；若正常推理阶段未出现已应用轮次，"
        "该示例不足以证明持续在线 Patching 的效果。",
        "",
        "自动核验通过：输出逐 token 等于静态基线；三个请求持续活跃；"
        "缩扩容保留活跃 block 映射且无抢占；实际 KV 分配量缩小并恢复；"
        "独立校验运行中源端与目标端全量迁移 KV 的 SHA-256 相同。",
        "",
        "指标口径：暂停包括流水线排空和最终切换；NCCL 时间来自源端 CUDA "
        "send event，不含 CPU 握手和打包；Forward 窗口包含 kernel 间隙。"
        "本实验的出 token 时间线证明后台迁移期间请求继续推进，"
        "同时执行的 GPU kernel 比例需要独立 CUPTI 实验。"
        "KV 分配量为存活 stack tensor 的大小，PyTorch reserved 显存可能保持不变。",
        "",
        "容器使用现有 NCCL SHM 重定向和 CPU ring 大小适配，"
        "启动参数见 ../server.log；复现实验命令见仓库 docs/design/pipelive.md。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
