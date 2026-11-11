# 动态 PP 调度器实验报告

数据来源：双 GPU 实际 HTTP 推理与 NCCL 测量；原始记录位于本目录。

本次每个策略仅运行一轮，结果不提供统计显著性或跨负载收益保证。

## 对照结果

每个策略使用相同开放环到达序列：0/3/6/9 秒分别提交 1/4/8/1 个请求，上下文 32/128/64/32 Token，各生成 256 Token。固定 seed=42、temperature=0，每组重置前缀缓存并预热；多轮运行时反转执行顺序。

|策略|重复数|吞吐 Token/s 均值|TPOT ms 均值|SLO Goodput Token/s|迁移次数合计|
|---|---:|---:|---:|---:|---:|
|full|1|76.38|82.27|43.65|1|
|static_12|1|69.37|90.07|39.64|0|
|static_16|1|81.42|74.46|46.52|0|
|static_20|1|68.55|91.06|39.17|0|
|static_24|1|59.37|107.28|33.93|0|
|static_28|1|53.22|120.77|30.41|0|
|static_4|1|54.60|117.74|31.20|0|
|static_8|1|61.22|103.66|34.99|0|
|threshold|1|58.36|109.80|33.35|1|

Best Static（本测试所有合法固定边界的事后最优值）为 static_16。SLO 定义为 TTFT≤5000ms 且 TPOT≤150ms。

## 资源与迁移成本

|策略|抢占增量合计|KV 压力峰值|迁移总耗时 ms|暂停总耗时 ms|
|---|---:|---:|---:|---:|
|full|0|0.976|6698.68|201.28|
|static_12|0|0.913|0.00|0.00|
|static_16|0|0.976|0.00|0.00|
|static_20|0|0.976|0.00|0.00|
|static_24|0|0.976|0.00|0.00|
|static_28|0|1.000|0.00|0.00|
|static_4|0|0.976|0.00|0.00|
|static_8|0|0.976|0.00|0.00|
|threshold|0|1.000|3486.64|193.03|

## 在线更新与自动迁移

稳定尾段有 28 个 Compute 桶获得多次更新；初值、终值、样本数和误差保存在 analysis.json 的 online_convergence。

独立性能变化实验：1 次自动迁移，事务合计 8309.21ms，暂停合计 401.97ms。

## 预测和正确性

逐样本先预测后更新的误差：`{"decode": {"count": 1146, "mean_absolute_relative_error": 0.015380512642048288, "p95_relative_error": 0.040301532001954424}, "prefill": {"count": 134, "mean_absolute_relative_error": 0.01400017220787056, "p95_relative_error": 0.04308798933034561}}`。首个无模型样本不计入误差。这是在线预序误差，不是独立数据集泛化精度。

相同上下文的逐 Token 不一致记录：0。完整请求计数、抢占增量、KV 峰值、迁移总耗时和暂停耗时见 `analysis.json`。

## 解释与边界

当前使用两张 RTX 3090 Ti 与 llama7b 校验端到端行为。历史 Llama-7B 迁移数据保存为独立 historical_profile.json，按模型、dtype 与环境条件隔离，不直接混入当前计算分桶。

模型使用固定开销＋每层开销曲线，并用分桶 EWMA 更新速度倍率。CUDA 事件从 V2 的 PP 接收完成后开始计时，包含模型执行与主机发射间隙，不能当作纯算子时间。小消息网络观测包含主机 staging，迁移通信统计提供大消息有效带宽。冷启动未覆盖的工作负载保持当前分层。空载迁移的干扰值采用 1.0 保守上界，在线迁移后以实际执行步数更新。

security_mode=off。可信执行、加密、证明验证器均未实现；可信策略必须提供有效节点能力且需要实际安全成本提供器才能参与决策。

## 实现与复现入口

实现分支：`feat/adaptive-pp-scheduler`。运行与配置见 [README](README.md)，
五个独立脚本由 [run_all.sh](run_all.sh) 使用 nohup 顺序执行。
默认对照运行三轮；本次最终硬件验证使用 `REPEATS=1`。

最终完整策略吞吐 76.38 Token/s，高于阈值策略的 58.36，低于 Best Static 的 81.42。
这些数据证明采样、更新、自动决策与在线迁移链路可运行，不证明动态策略普遍优于固定分层。

原始数据保留在仓库的 `benchmark_results/adaptive-final-llama7b/`：
`profile.json`、`network.json`、`profile_samples.jsonl`、`profile_migrations.jsonl`、
`*_requests.jsonl`、`*_samples.jsonl`、各策略的 `decision.log`/`outcome.log`、
`migrations/`、`analysis.json` 和五个脚本日志。该目录约 186MiB，沿用仓库结果目录的忽略规则。
可用 `.venv/bin/python experiments/adaptive_pp/report.py --out benchmark_results/adaptive-final-llama7b`
从原始数据重新生成结果报告。

`tests/v1/worker/test_adaptive_pp.py` 与 `test_pp_reconfigure.py` 共 126 项通过；
修改文件的 pre-commit 检查通过。日志位于上述结果目录的 `unit-tests.log` 和 `precommit.log`。

## 服务启动集成验证

以 `VLLM_PP_ADAPTIVE_PROFILE` 启动服务，使用 `--server-managed` 仅发送负载。
内置调度器自动从 20/12 调整到 16/16；迁移总耗时 3156.22ms，暂停 162.82ms。
14 个请求全部完成，共 3584 Token，与固定 16/16 的确定性输出逐 Token 比较无差异。
验证结束时调度器仍运行、失败锁止为 false，服务无剩余请求，配置版本为 1。
原始状态、输出和校验结果见 `benchmark_results/adaptive-startup/` 的
`startup_status.json`、`scheduler_requests.jsonl`、`validation.json`、`runtime/` 和 `migrations/`。
实验服务已在请求完成后停止。
