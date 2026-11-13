# 固定异构 GPU 的静态 PP 切分搜索

在仓库根目录执行。结果根目录：`/workspace/results/heterogeneous_pp_split/`。
正式搜索必须手动启动；本次准备阶段只执行标定、短测及回归。

```bash
cd /workspace/molink-runtime/vllm
# 启动：内部使用 nohup，关闭终端后继续；自动跳过已有终态组
.venv/bin/python experiments/heterogeneous_pp_split/control.py start
# 进度、存活状态、成功/失败数
.venv/bin/python experiments/heterogeneous_pp_split/control.py status
# 实时总日志（每组详细 server.log 路径见进度对应的结果目录）
tail -F /workspace/results/heterogeneous_pp_split/search.log
# 停止并恢复；保留结果供续跑
.venv/bin/python experiments/heterogeneous_pp_split/control.py stop
# 汇总结果与图表
cat /workspace/results/heterogeneous_pp_split/best.json
ls /workspace/results/heterogeneous_pp_split/{summary.csv,curves.png}
```

`config.json` 是统一配置。搜索固定 17 个 k × 3 类负载；每组独立重启服务，
预热发压 30 秒后排空，测量发压 300 秒后排空。
输入复用已有 `adaptive_pp/experiment.py::generate` 的 `[42] * input`，
seed=42、temperature=0、ignore_eos=true；所有切分相同。每个并发槽完成后立即补请求。
Chunked Prefill 开启、Prefix Cache 关闭；FP16，PP=2、TP=1，eager，TRITON_ATTN。
保留 MoLink 只读监控所需开关，自动调度未启用、在线迁移关闭，Gate 在前后均断言关闭。
每个整数切分都通过 `VLLM_PP_LAYER_PARTITION` 静态启动，不调用迁移接口。

时钟控制被容器权限拒绝，替代方式为私有 CUDA MPS、GPU1 活跃线程上限、
GPU0 真实显存复制竞争及显存分配预留。不是时间延迟注入。
MPS 线程比例不等于 GEMM 性能比例；以 `calibration.json` 的实测为准。
配置固定的是资源限制和竞争占空比，无法锁定 Boost 时钟或保证所有负载下恒定性能比。
GPU0/GPU1 保留约 12/20 GiB 空闲物理显存，服务预算为 11/19 GiB，剩余用于 CUDA/NCCL。
worker 扩展只在实验目录内设置每卡预算并补充监控，不修改 MoLink 核心。
PP 共用逻辑 block 数量，因此较大显存一侧不一定全部分配给 KV。

吞吐：300 秒窗口内收到的输出 token / 300；尾部排空 token 不计入窗口吞吐。
Goodput：上述窗口 token 中，其完整请求同时满足 TTFT≤5000ms、TPOT≤100ms 的部分。
TTFT 从 HTTP 发出计时，包含排队；TPOT=(最后 token−首 token)/(输出数−1)。
P95 使用 nearest-rank。SLO 为本实验显式假设，可在重新标定/短测前修改配置。
逐请求时间戳及 token IDs 均保留。客户端可见流式时间可能受 HTTP 分块影响。

Stage 耗时复用 MoLink CUDA Event，范围为 PP 接收完成后的 forward；
按 sequence 去重，分别记录 Prefill/Decode 时间。KV 为 block 分配镜像；
峰值包含 worker allocator 高水位和采样到的整卡显存峰值（后者包含显存预留）。
NVML GPU 利用率包含竞争负载，不能单独证明推理饱和。
流水线气泡与切分不均导致的空闲需要结合 Stage 时间和在途请求解读。

每次尝试使用独立时间戳子目录，保存启动参数、请求、采样、服务日志和结果。
`search/<负载>-k<层数>/result.json` 为该组终态索引；失败记录原因及日志路径。
启动/OOM/请求错误保留失败记录，不把失败吞吐填成零。停止中的组没有终态，续跑重试。
已记录失败也会跳过；如明确需要重试，保留原时间戳目录，仅移动该组终态索引后重启。

后台 supervisor 在 runner 异常退出时清理已记录进程和私有 MPS。
正常退出、异常及 TERM/INT 都释放预留。`stop` 也可恢复 runner 被 SIGKILL 后的状态。
时钟设置从未成功修改，无需请求驱动重置。不要使用 `kill -9` 同时杀掉 supervisor 和 runner；
若这么做，用上面的 stop 命令恢复。系统断电后 GPU 进程资源由驱动回收。

准备阶段复现（均为短测，不启动正式搜索）：

```bash
.venv/bin/python experiments/heterogeneous_pp_split/run.py calibrate
.venv/bin/python experiments/heterogeneous_pp_split/run.py pilot
.venv/bin/python experiments/heterogeneous_pp_split/run.py smoke --smoke-split 15
.venv/bin/python -m pytest --confcutdir=experiments/heterogeneous_pp_split experiments/heterogeneous_pp_split/test_metrics.py -q
.venv/bin/python experiments/heterogeneous_pp_split/freeze.py
```

`freeze.py` 核对标定、实际 84/48 SM、三类瓶颈与请求供给记录，再冻结配置和实验源码哈希。
正式脚本拒绝混用更改后的配置/源码。Pilot 各组发压 2 秒预热、20 秒测量，
但会排空 1024/512-token 请求，因此总用时大于发压窗口。Smoke 为 2+5 秒发压。
硬件标定和旧尝试保留在各自时间戳目录。

完整搜索至少需要 51×330 秒（4 小时 40 分 30 秒），另加模型启动与请求排空。
最终除 `best.json`、`summary.csv`、`curves.png`，还会生成 `resources.png` 和 `report.md`。
`best.json` 包含两卡 Stage、KV、显存详细指标；吞吐与 Goodput 分别选优，并保留并列项。
`report.md` 判断三类负载是否存在共同最优切分；基线失败或为零时不计算百分比提升。

验收记录见结果目录 `preparation_report.md`、`unit-tests.log` 和 `lifecycle-tests.json`。
交接时最后一次运行状态为有意 SIGKILL 的 smoke 恢复测试；
`formal_search_started=false`、0/51 表示正式搜索尚未开始。
完整搜索的完成依据是 `status.mode=search`、`status.status=completed`、51 组终态及 `completed` 文件。

搜索完成后，可在当前 VS Code 环境直接打开结果与图表：

```bash
code /workspace/results/heterogeneous_pp_split/report.md /workspace/results/heterogeneous_pp_split/curves.png /workspace/results/heterogeneous_pp_split/resources.png
```
