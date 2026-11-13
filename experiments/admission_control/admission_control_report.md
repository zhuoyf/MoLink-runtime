# Migration Admission Control 实现与验证

验证日期：2026-11-13（北京时间）。本次在已有未提交实现上继续完成，保留原有实验数据；
本轮数据独立保存在 `results/validation-20261113/`。

## 实现

V1 Scheduler 继续使用原有 WAITING/RUNNING 队列。在 WAITING 请求进入执行前检查
独立 `migration_admission_gate`；开启时记录已运行、已抢占或已有计算进度的请求，
允许它们继续 Decode、Chunked Prefill 或抢占后恢复。新请求留在等待队列，取消仍走
原有请求清理路径。Gate 不修改 PauseState，因此 Commit 后 Resume 不会提前放行。

EngineCore 提供 `molink_admission_gate` 的 open/close/status/renew RPC；owner 防止
其他事务解除租约。默认等待期限 10 秒，使用 monotonic clock；Scheduler 和状态查询
都会检查租约到期，前端退出后不依赖前端 finally 才能恢复准入。EngineCore 自身退出时
请求无法继续服务，需使用现有进程重启机制；新进程 Gate 初始关闭。

自适应调度器执行 `NORMAL → PREPARING → RECLAIMING → MIGRATING → RECOVERING → NORMAL`。
初次物理容量预检不可行但存在收益时开启 Gate；每 250ms 重新检查租约、配置版本、
资源状态和所有合法 stacking 边界的收益，可改选移动层数更少的目标。收益消失、
停止、超时、配置变化和异常均通过 finally 解除 Gate。异步锁防止同一调度器的并发尝试，
控制器锁与 expected_version/expected_boundary 防止并发重配置及过期决策提交。

空间判断复用 worker `describe` 提供的中间峰值/最终容量及 device/allocator 空闲信息，
并用 `plan_resize` 检查 null block 和引用中的 KV；预检本身不删除 block。
执行器对所有 worker 完成真实 KV Shrink 后再次 describe 检查峰值容量，再进入在线迁移。
失败时走现有回滚，并尝试恢复缩容前容量；恢复失败保留事件记录。
仅逻辑空闲增加不足以通过物理容量检查。

进入 MIGRATING 时租约延长至 480 秒，覆盖现有执行器 180/180/60 秒的阶段等待上限；
准入等待本身仍由默认 10 秒的 asyncio.timeout 限制。租约是独立的失效保护，
不替代现有执行器对 worker/通信故障的处理。

本轮补充将收益重评估的排队代价改为 EngineCore 积分得到的 `queue_seconds`，
避免按瞬时 Waiting 数丢失已发生的排队时间。未开启 Gate 的预测仍用等待预算与当前队列估计。
请求计时从 HTTP 发送开始，包含排队；最终 Goodput 同时检查 TTFT 与 TPOT，
吞吐分母包含负载结束后的排空时间。

## 修改文件

| 文件 | 用途 |
|---|---|
| `vllm/distributed/pp_reconfigure/admission.py` | 独立 Gate、owner、租约、排队时间 |
| `vllm/v1/core/sched/scheduler.py` | WAITING 准入拦截与已有请求恢复 |
| `vllm/v1/engine/core.py` | Gate RPC、物理容量预检、状态输出 |
| `vllm/distributed/pp_reconfigure/adaptive/runtime.py` | 状态机、轮询、清理与收益重评估 |
| `vllm/distributed/pp_reconfigure/adaptive/optimizer.py` | 回收候选、等待/排队成本及配置 |
| `vllm/distributed/pp_reconfigure/control.py` | 缩容后物理峰值复检、失败容量恢复 |
| `vllm/entrypoints/serve/dev/molink/api_router.py` | 开发 HTTP 接口 |
| `experiments/adaptive_pp/experiment.py` | HTTP 测试客户端调用 |
| `tests/v1/core/test_scheduler.py` | Decode、抢占恢复、Pause 独立与取消 |
| `tests/v1/worker/test_adaptive_pp.py` | 租约、收益、较小目标及故障恢复 |
| `tests/v1/worker/test_pp_reconfigure.py` | 物理容量与引用 block 预检 |
| `experiments/admission_control/` | 五个独立脚本、负载驱动、服务启动、文档与原始数据 |

## 实验方法

两张 RTX 3090 Ti，vLLM 0.29.0+molink.cu129，tiny Llama，PP=2、TP=1、BF16、eager。
初始分区 6/4，192 KV blocks，最大请求并发 64，关闭 Prefix Cache，
GPU memory utilization=0.06，以约 1.42 GiB/卡的服务预算构造受限容量。
该模型每层每 block 物理 KV 页为 1 MiB。此测试未将整张 GPU 填满至驱动 OOM。

输入/输出各 128 tokens、temperature=0、seed=42，每个并发槽位完成后立即补充请求。
Gate/回收/故障用例负载时长 5 秒，自动迁移 20 秒，比较实验 30 秒，均额外排空在途请求。
自动/比较实验在 stage 0 注入 3ms/层 Decode 延迟，使用匹配的已有 profile，horizon=300 秒。
SLO：TTFT ≤ 5000ms 且 TPOT ≤ 100ms。每组一次运行；结果用于机制回归和此次配置下的观察。
功能实验期间同时运行 CPU 单元测试；比较实验的运行环境与时间可由原始日志追溯。

复现命令与参数覆盖方式见 [README.md](README.md)。逐请求数据、Token 时间戳、模型、
决策、状态机、初始/最终状态与采样均保存；server-current.log 保存服务端输出，
`results/migrations/` 保存每次迁移各阶段日志，outcome 中包含对应目录。

## 测试与性能结果

相关回归：**262 passed，42 deselected**。42 项按类别排除，其中包括完整运行中的 37 项失败及 5 项同组测试。
完整三文件运行：266 passed、37 failed；36 项因当前 Transformers 缺少 `PixtralRotaryEmbedding` 导致 Llava 检查失败，1 项现有 fixture 缺少 `_tpc_pred`。用 HEAD Scheduler 在独立进程复现两类代表失败，见 `baseline-tests.log`。最终回归另新增了较小目标测试。Ruff 与 `git diff --check` 通过。

五个脚本共完成 18 个 GPU 用例，另复测并发 16 的两组比较，共 20 个用例。全部请求 Token 与各自基准一致，最终 Gate 关闭、队列排空、used_blocks=1（null block）。详见 `audit.json`。

| 用例 | 并发 | 结果 |
|---|---|---|
| Gate | 16 / 32 / 64 | 新请求等待、已有请求持续 Decode，恢复准入后排空 |
| 回收 | 16 / 32 / 64 | 192→96 blocks，合计释放 960 MiB KV 分配，物理容量预检通过 |
| 故障 | 16 / 32 / 64 | 租约过期、非法边界拒绝、等待请求取消及最终恢复通过 |
| 自动迁移 | 16 / 32 / 64 | 均经历完整状态机，192→162 blocks，释放 300 MiB，迁移成功 |

单元测试还覆盖容量等待超时、RPC 无响应、Gate 开启响应丢失、控制任务取消、版本变化、收益消失、执行器异常、日志异常、租约 owner、较小目标重评估及真实 Scheduler 抢占恢复。

自动迁移结果：

| 并发 | Gate 采样秒 | Waiting 峰值 | 迁移 ms | Pause ms | TTFT ms | TPOT ms | 吞吐 token/s | Goodput token/s |
|---|---|---|---|---|---|---|---|---|
| 16 | 5.99 | 12 | 1048.4 | 11.6 | 696.2 | 41.2 | 312.0 | 312.0 |
| 32 | 6.33 | 23 | 997.3 | 110.0 | 4459.4 | 46.7 | 341.3 | 182.0 |
| 64 | 6.09 | 55 | 1039.6 | 116.8 | 10324.8 | 53.6 | 356.7 | 70.0 |

性能比较（并发 16 使用单元测试结束后的独立复测，其余使用主实验；原始两次数据均保留）：

| 并发 | Gate | 最终分区 | Gate 秒 | Waiting 峰值 | 迁移成功/尝试 | 迁移 ms | Pause ms | TTFT ms | TPOT ms | 吞吐 token/s | Goodput token/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 关 | 4/6 | 0.00 | 7 | 1/1 | 1051.1 | 112.0 | 196.0 | 48.3 | 310.6 | 310.6 |
| 16 | 开 | 2/8 | 7.01 | 12 | 1/1 | 1098.6 | 109.2 | 641.1 | 41.2 | 321.3 | 321.3 |
| 32 | 关 | 4/6 | 0.00 | 21 | 1/1 | 936.9 | 120.0 | 4601.5 | 50.1 | 326.1 | 106.7 |
| 32 | 开 | 2/8 | 5.98 | 23 | 1/1 | 987.6 | 16.7 | 4772.4 | 43.0 | 355.0 | 167.8 |
| 64 | 关 | 4/6 | 0.00 | 54 | 1/1 | 955.2 | 54.8 | 13147.5 | 49.8 | 331.2 | 50.0 |
| 64 | 开 | 2/8 | 5.97 | 55 | 1/1 | 1118.6 | 107.0 | 12038.8 | 49.9 | 370.2 | 55.1 |

本轮 Goodput 相对变化：

- 并发 16：+3.4%；TTFT 196.0→641.1ms。
- 并发 32：+57.3%；TTFT 4601.5→4772.4ms。
- 并发 64：+10.2%；TTFT 13147.5→12038.8ms。

关闭 Gate 的三组也能迁移，但选择的是较保守的 4/6；开启 Gate 的三组可回收后执行 2/8。因此此次结果证明扩大了可执行迁移目标范围，并非总体迁移成功率从零提升。启用组较大的移动需释放 300 MiB；关闭组无需缩容。成功率两边均为 1/1，不能用此样本宣称普适的成功率提升。

Gate 带来显式排队，TTFT 可能增加；并发 64 下绝对 Goodput 仍受 5 秒 TTFT SLO 严重限制。收益模型中的 horizon=300 秒与 30 秒观察窗不同，短窗口数据必须连同排空成本解读。

局限：tiny 模型、注入异构延迟、预算约束显存、单次配对且固定先关闭后开启，无法给出统计显著性或生产大模型结论；未执行整卡显存耗尽、长时间 soak 或硬件/EngineCore 崩溃后的进程重启测试。逻辑引用和 Token 一致性检查通过，不等价于长期显存泄漏证明。

数据索引：`summary.jsonl` 原始汇总、`summary.csv` 两轮汇总、`audit.json` 状态检查、各用例 `requests.jsonl`/`samples.jsonl`/`parameters.jsonl`、`commands.txt`、`packages.txt`、`gpu-environment.txt`、`git-head.txt`、`working-tree.patch`。历史结果没有覆盖，未用于本报告的比较表。
