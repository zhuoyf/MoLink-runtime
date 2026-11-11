# 在线 PP 分层调度

实现位于 `vllm/distributed/pp_reconfigure/adaptive/`，通过现有
`ReconfigurationController.reconfigure(..., mode="async")` 执行迁移。
PipeLive 的复制、Patching、暂停、提交、回滚和请求/KV 一致性机制沿用原实现。
控制器仅增加锁内的版本与当前边界检查。

## 运行

从仓库根目录运行，使用已安装 vLLM 的 `.venv`：

```bash
bash experiments/adaptive_pp/run_all.sh
# 返回后台 PID；五个实验按顺序运行。
tail -f benchmark_results/adaptive/run_all.log
```

每个 `01_profile.sh` 至 `05_security_interface.sh` 也可单独执行。
`OUT` 指定结果目录，`URL` 指定已启动服务，`MODEL` 指定本地 Llama 权重。
默认实验使用现有 `/workspace/model/molink-tiny-llama`、PP2/TP1、BF16、
stack=2、128 个 KV blocks。修改模型时，同时调整 `serve.sh` 的资源参数。
两张 GPU 需要空闲；实验脚本仅启动自己的服务，不停止已有服务。

容器的 `/dev/shm` 只有 64MiB，`common.sh` 默认复用既有 PipeLive 的
`nccl_shm_redirect.so`（如存在）。`SHM_PRELOAD` 可指定该环境辅助库；
正常大共享内存环境可设为空。NCCL 配置必须在 Profiling 与服务端保持一致。
独立网络校准使用 NCCL ping-pong，不在推理热路径上加入大消息探测。

`01_profile.sh` 先校准网络，再对所有合法边界采样计算和 KV，测量双向迁移。
相同条件的 `profile.json` 自动复用；`01_profile.sh --force` 显式重测。
原有 Llama-7B 的 10 次 Full Online 迁移导入独立的
`historical_profile.json`，保存来源 SHA256、负载和环境；不混入小模型分桶。

`02_online_update.sh` 在动态负载及八轮稳定尾段下更新模型，保留各 bucket 的更新轨迹、样本数和误差；
`03_scheduler.sh` 从不均衡分层启动并添加 20 秒 GPU0 矩阵计算负载，验证性能变化下的决策；
`04_comparison.sh` 对全部固定边界、阈值策略和完整策略各执行三次。
请求在 0/3/6/9 秒按固定序列到达，不因上一批完成时间改变到达计划。
每轮独立复制初始模型，重置前缀缓存并预热，第二轮反转策略顺序。
Best Static 是所有合法固定边界在测试集上的事后最优值。
`05_security_interface.sh` 检查默认关闭、能力缺失、过期证明与成本提供器缺失。

## 随服务启动

完成 Profiling 后，在启动服务前配置：

```bash
export VLLM_PP_ADAPTIVE_PROFILE="$PWD/benchmark_results/adaptive/profile.json"
export VLLM_PP_ADAPTIVE_LOG_DIR="$PWD/benchmark_results/adaptive/production"
export VLLM_PP_ADAPTIVE_ALPHA=0.2
export VLLM_PP_ADAPTIVE_INTERVAL=5
export HETVLLM_RESOURCE_MONITOR_INTERVAL=1
export VLLM_PP_ADAPTIVE_CONFIG='{"horizon":60,"cooldown":30,"min_gain":0.05}'
bash experiments/adaptive_pp/serve.sh
```

该服务沿用 MoLink 开发控制接口和 Llama/eager/PP2 的现有限制。
`GET /molink/adaptive/status` 返回状态、最近决策和模型分桶数量。
模型配置摘要、dtype 和 GPU UUID 不匹配会拒绝使用旧 Profiling。
启动缺少 Profiling 文件时直接报错；独立预处理是支持的初始化路径。
不要同时运行服务内调度器与实验脚本的外部调度器。
停止服务会等待当前迁移完成；事务失败后自动迁移保持禁用，需要检查原日志并重启。

## 模型和决策

- 节点复用非阻塞 CUDA 事件和本地采样线程。NCCL 汇总沿用已有有序 RPC。
  原始 Compute 样本保留 phase、batch、context、tokens、layers、时间和序号。
- Compute 以节点和 2 的幂工作负载桶维护非负的固定开销＋每层开销模型。
  Profiling 提供初始曲线，在线 EWMA 更新同桶的速度倍率；不同工作负载不直接混合。
  V2 的 CUDA 起始事件在 PP 接收完成之后记录，排除上游等待。
- KV 使用实际分配的每层每 block 字节数，包含 Layer Stacking 分配粒度。
  预测容量遵守配置的 blocks 上限；峰值按迁移时两端旧/新层集合的并集计算，
  加权重与 128MiB 迁移缓冲。当前策略保守拒绝需要先缩容才能容纳的候选。
- 小消息 NCCL 汇总耗时更新网络延迟；迁移时实测通信字节/时间更新有效带宽。
  Compute、KV、网络、迁移耗时和干扰分别记录，空载迁移干扰用 1.0 上界。
- 使用 KV 增长 EWMA 与排队请求的上下文预测窗口内 block 需求；超出容量的
  比例作为抢占风险代理。它是代价估计，不声称是校准后的抢占概率。
- 多请求 Decode 使用 PP 最慢阶段的吞吐成本；单请求和 Prefill 使用阶段延迟之和。
- 在同一 horizon 内，以当前阶段服务速率归一化比较服务秒数：
  `T*(1+current_risk) + (H-T)*(candidate_step/current_step+risk)`，
  再加窗口内暂停与干扰成本，`T=min(H,migration_total)`；迁移完成前按旧分层计费。
  所有项单位为秒。计算误差会提高收益门槛；收益不足保持当前配置。
- 枚举所有 stack 对齐的内部分界。缺失分桶、过期/异常采样、显存不足、
  Cooldown 或版本不匹配时保持配置。迁移失败锁止自动迁移。
- 决策和文件写入在 API 后台任务/线程执行。只复用现有轻量监控 RPC；
  不在 worker 的推理循环中运行优化器或文件写入。

`decision.log` 包含 RuntimeState、版本、所有候选、容量/峰值、预测代价和原因；
`outcome.log` 包含实际迁移结果及对应决策；`sample_error.log` 记录缺失/过期数据。
模型 JSON 保存各桶的值、最后更新时间、样本数及更新前相对预测误差。

## 安全接口

参考 `/workspace/HydraInfer.pdf` 第 IV 节的 Edge/Cloud 与 TEE/REE 联合放置思想，
将放置可行性与开销计算分离：`PlacementPlan`、`SecurityPolicy`、
`SecurityCapabilities`、`SecurityCostModel`。
当前只运行 `security_mode=off`；不实现论文中的混淆、加密或可信执行算法。
可信模式要求两节点的 TEE、可信 GPU、验证成功且未过期的证明；缺任一项拒绝。
即便能力满足，默认安全成本提供器仍拒绝可信模式，直到接入真实实现。
当前随服务配置非 `off` 模式会直接拒绝启动，避免继续以 REE 执行安全请求。

## 验证与产物

```bash
.venv/bin/python -m pytest tests/v1/worker/test_adaptive_pp.py tests/v1/worker/test_pp_reconfigure.py -q
.venv/bin/python experiments/adaptive_pp/report.py --out benchmark_results/adaptive
```

生成 `scheduler_report.md`、`analysis.json`、模型、网络校准、逐请求 Token/时间、
节点快照与决策/迁移日志。报告保留零收益及失败结果，不把小模型吞吐外推为生产收益。

## Llama-7B 复现配置

```bash
MODEL=/workspace/model/llama7b GPU_MEMORY_UTILIZATION=0.80 \
VLLM_PP_LAYER_PARTITION=20,12 VLLM_PP_RECONFIGURE_STACK_SIZE=4 \
OUT="$PWD/benchmark_results/adaptive-llama7b" \
bash experiments/adaptive_pp/run_all.sh
```

应先停止本实验先前启动的小模型服务，避免 `URL` 指向错误模型。
随服务启动的验证可在设置 `VLLM_PP_ADAPTIVE_PROFILE` 并启动服务后执行：

```bash
.venv/bin/python experiments/adaptive_pp/experiment.py scheduler \
  --server-managed --out benchmark_results/adaptive-startup --stack 4 \
  --model /workspace/model/llama7b
```

该模式仅提交负载和检查 `/molink/adaptive/status`，不启动外部调度器。
