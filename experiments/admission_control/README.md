# Migration Admission Control 实验

从仓库根目录启动双 GPU 服务：

```bash
bash experiments/admission_control/serve.sh > /tmp/admission-server.log 2>&1
```

另一个终端依次执行；每个脚本默认并发 16、32、64，持续补充请求，
结束负载后等待在途请求完成。输出目录中的同名实验必须不存在，避免混入旧日志。

```bash
bash experiments/admission_control/01_gate.sh --out /tmp/admission-run
bash experiments/admission_control/02_memory_reclaim.sh --out /tmp/admission-run
bash experiments/admission_control/03_auto_migration.sh --out /tmp/admission-run
bash experiments/admission_control/04_failure_recovery.sh --out /tmp/admission-run
bash experiments/admission_control/05_comparison.sh --out /tmp/admission-run
```

公共参数：`--url`、`--profile`、`--concurrency 16 32 64`、`--seconds 30`、
`--repeats 1`、`--context 128`、`--tokens 128`、`--ttft-slo 5000`、
`--tpot-slo 100`。时间单位：SLO 使用毫秒，负载时长使用秒。
比较实验对每个并发分别运行 Gate 关闭、开启两组。

默认使用 `/workspace/model/molink-tiny-llama`、PP=2、192 个 KV blocks、
6/4 初始分区和预算受限的真实 GPU KV 分配。自动迁移实验在 stage 0 注入
每层 3ms Decode 延迟以产生可重复的迁移收益；属于机制回归实验。
`MODEL`、`PORT`、`GPU_MEMORY_UTILIZATION` 可通过环境变量覆盖。
更换模型必须重新生成匹配的性能 profile。

每个实验保存参数、初始/最终状态、逐请求 Token ID/TTFT/TPOT、周期采样、
预测模型、决策及迁移结果。`summary.jsonl` 汇总吞吐、SLO Goodput、
迁移成功率/耗时/Pause、Gate 采样时长、Waiting 峰值、KV 分配释放量。
吞吐分母包含排队及结束后的排空时间，未满足任一 SLO 的请求不计入 Goodput。
Gate 时长由 100ms 周期采样近似；精确租约时间可查原始 Gate 状态。
KV 释放量表示 GPU KV 物理分配减少，不等同于 CUDA allocator 向驱动归还的显存。

Gate 超时默认 10 秒，可在生产调度器配置中设置：

```bash
export VLLM_PP_ADAPTIVE_CONFIG='{"admission_control":true,"admission_timeout":10,"admission_poll_interval":0.25}'
```

EngineCore RPC：`molink_admission_gate("open", owner, timeout)`、
`molink_admission_gate("close", owner)`、`molink_admission_gate()`。
`open` 表示开启准入限制；`close` 表示解除限制。开发 HTTP 接口对应
`POST /molink/admission` 和 `GET /molink/admission`。

故障脚本验证租约过期、非法迁移和 HTTP 请求取消；控制器取消、容量超时、
RPC 超时、配置变化、收益消失、迁移异常和日志异常通过单元测试注入。

```bash
.venv/bin/python -m pytest tests/v1/worker/test_adaptive_pp.py tests/v1/worker/test_pp_reconfigure.py tests/v1/core/test_scheduler.py -q
```
