# 动态异构 GPU 实验

基于现有 PP2、Layer Stacking=4、Llama-7B BF16 和 PipeLive 在线迁移。
代码分支 `feat/dynamic-heterogeneous-benchmark`。

## 模拟与约束

`HeterogeneitySimulator` 在 V2/eager 的 PP 接收完成、模型执行之前按当前层数
注入 Decode 主机延迟。延迟不持有 NCCL 或 KV 迁移锁，CUDA 事件与端到端时间
记录实际影响；这是一种可控服务时间模拟，不等价于真实 GPU 时钟降频。

开发接口 `POST /molink/heterogeneity` 仅在
`MOLINK_HETEROGENEITY_SIMULATION=1` 时可用，接受两个节点的
`decode_ms_per_layer` 和 `kv_budget_bytes`。KV 字节预算换算为各阶段的 Blocks
上限；共同逻辑池取两者最小值，使用已有 resize 事务实际释放/分配 KV。
因此不会把只改变监控数字当作容量限制。迁移前按旧/新层集合的最大值预缩容，
迁移后重新分配；禁止释放活跃请求 Blocks。显存峰值仍经过现有优化器检查。

所有策略都使用上述物理约束。`compute_only` 只去掉 KV 风险代价，
`without_compute` 去掉候选计算差异，`without_migration_cost` 去掉预测迁移代价；
它们均保留安全可行性检查、Cooldown 和实际迁移成本。

## 执行流程

从仓库根目录启动专用实验服务：

```bash
bash experiments/heterogeneous_benchmark/serve.sh
```

另一个终端依次执行预实验，需先完成环境中的 vLLM 安装：

```bash
bash experiments/heterogeneous_benchmark/01_profile.sh
bash experiments/heterogeneous_benchmark/02_validate.sh
```

第一个脚本实测全部 7 个合法切分点在 A/B/C 阶段的性能；预实验文件保留，
重复调用复用已完成的结果。第二个脚本运行 60 秒/阶段的在线调度验证，
检查两个方向的自动迁移、Token 正确性、Prefill 数量和物理 KV 一致性，
通过后生成 `frozen.json`、`frozen.sha256`。已存在的冻结文件不会被覆盖。
若预实验需重做，先把旧 `pilot/` 整体归档，保留全部历史证据。

正式实验采用 120 秒/阶段，7 个 Static、Threshold、Full、Oracle、
3 个消融及 `full_start28` 共 14 组，各运行 5 次；按轮次轮转并反转顺序。
所有策略重置缓存、执行相同预热，从同一性能模型副本开始。
请求开放环到达，跨阶段请求不取消，尾部排空时间计入总体指标。

均衡阶段性能接近时，单次 Profiling 的最佳点可能受波动影响，
必须结合正式固定策略的五次重复判断。Full 初始边界为 16，
`full_start28` 测量初始化的影响；Oracle 从相同边界开始并承担实际迁移成本。
报告另列忽略响应和迁移的阶段性能经验上界，它不属于实测策略。

预实验通过后先正常停止专用服务，再启动正式后台任务：

```bash
mkdir -p experiments/heterogeneous_benchmark/logs
nohup bash experiments/heterogeneous_benchmark/run_all.sh \
  > experiments/heterogeneous_benchmark/logs/run.log 2>&1 < /dev/null &
echo $! > experiments/heterogeneous_benchmark/run.pid
bash experiments/heterogeneous_benchmark/status.sh
```

`run_all.sh` 验证冻结输入、启动自己的服务，顺序执行正式对照和报告脚本。
原始逐请求 Token/时间、阶段应用时间、快照、模型更新和决策写入 `results/`。
服务与迁移日志在 `logs/`，正常/失败退出均写入 `exit_code`、`completed`、`status`。
SIGKILL 或宿主异常无法执行退出 trap；PID 消失且无 exit_code 表示异常结束。
不会自动重启或覆盖正式结果。正式启动后无需 Agent 等待。

```bash
tail -f experiments/heterogeneous_benchmark/logs/03_comparison.log
cat experiments/heterogeneous_benchmark/report.md
```

报告和 PNG 图由实际记录生成，未完成时明确标注；不承诺正向收益。
所有迁移启用现有 KV 校验，统计 Prefill Token 和 resumed/preempted 计数，
验证是否出现隐式重算。计时包含校验开销，对全部迁移策略保持一致。
