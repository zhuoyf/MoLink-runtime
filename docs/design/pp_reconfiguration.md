# MoLink Runtime

基于 **vLLM 0.29.0+molink.cu129** 的动态 Pipeline Parallel 推理运行时。
当前动态重配置路径支持本地 Llama、PP=2、TP=1、V2 Model Runner、
BF16、eager 和 Triton Attention。CUDA 内核使用同版本预编译 wheel。

保留异构 GPU 状态采集、Profiling、在线成本更新与切分点选择；
权重迁移、非连续 Attention、Layer Stacking、KV Shrink/Expand、历史 KV、
Dirty Bitmap、增量 Patch、安全 Commit，以及显存预检、Admission Gate 和恢复机制。
TEE 只保留扩展接口，运行时安全模式为 `off`。

## 安装

Linux x86_64、Python 3.12、两张 CUDA GPU。使用 `uv`：

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python --no-deps -r requirements/runtime-cu129.txt
export VLLM_VERSION_OVERRIDE=0.29.0+molink.cu129
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
uv pip check --python .venv/bin/python
```

`requirements/runtime-cu129.txt` 固定运行时及构建依赖的完整版本集合；
其他平台依赖与原生构建文件保留上游结构。此版本的动态 PP 启动入口使用 Python 服务端。

## 启动

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve.sh /path/to/llama \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 128 \
  --num-gpu-blocks-override 128 --gpu-memory-utilization 0.8
```

默认监听 `127.0.0.1:8000`，启动脚本固定所需运行模式，其余参数透传给 `vllm serve`。
未配置自动调度器时，按初始切分持续静态推理，也可手动迁移。
服务启用开发控制接口；仅向受信任的管理网络开放。
64 MiB `/dev/shm` 容器可设置 `NCCL_BUFFSIZE=1048576`、
`NCCL_SHM_DISABLE=0`、`VLLM_MQ_MAX_CHUNK_BYTES_MB=1`。

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/path/to/llama","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
.venv/bin/python -m vllm.distributed.pp_reconfigure monitor --samples 1
.venv/bin/python -m vllm.distributed.pp_reconfigure reconfigure --boundary 18
```

`boundary=18` 表示 32 层模型使用 `18/14` 分层；必须是合法的 stack 对齐内部分界。
默认控制命令使用完整异步迁移，也可传 `--mode sync`。

| 参数 | 用途 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | 两张 GPU 的顺序 |
| `VLLM_PP_LAYER_PARTITION=16,16` | 初始分层；缺省由 vLLM 分配 |
| `VLLM_PP_RECONFIGURE_STACK_SIZE=2` | KV Layer Stacking 粒度 |
| `VLLM_PP_RECONFIGURE_LOG_DIR` | 迁移事务日志，默认 `runtime/migrations` |
| `HETVLLM_RESOURCE_MONITOR_LOG` | 节点监控日志，默认 `runtime/resources` |
| `VLLM_PP_ADAPTIVE_PROFILE` | 自动调度所需的本机、本模型 Profiling 文件 |
| `VLLM_PP_ADAPTIVE_CONFIG` | JSON 策略参数，如 `{"horizon":60,"cooldown":30,"min_gain":0.05}` |
| `VLLM_PP_ADAPTIVE_INTERVAL=5` | 自动决策间隔（秒） |
| `VLLM_PP_ADAPTIVE_LOG_DIR` | 调度日志，默认 `runtime/adaptive` |

## Profiling 与自动调度

先在空闲设备上校准通信，使用与服务相同的 GPU 顺序和 NCCL 环境：

```bash
mkdir -p runtime/profile
.venv/bin/python -m torch.distributed.run --nproc-per-node 2 \
  -m vllm.distributed.pp_reconfigure.network_profile \
  --output runtime/profile/network.json
```

然后启动专用于校准的服务（暂不设置 `VLLM_PP_ADAPTIVE_PROFILE`），运行：

```bash
.venv/bin/python -m vllm.distributed.pp_reconfigure.profile \
  --model /path/to/llama --stack 2 --out runtime/profile
```

校准会遍历合法切分、发送短请求并测量迁移，结束后恢复初始切分。
匹配的校准文件默认复用，`--force` 重采。完成后停止校准服务，使用同一启动命令，
增加 `VLLM_PP_ADAPTIVE_PROFILE="$PWD/runtime/profile/profile.json"`。
模型配置、dtype、stack 与 GPU UUID 必须匹配；缺少可信样本或收益不足时保持当前切分。
迁移异常后自动调度锁止，需要检查事务日志并重启。

管理接口：`GET /molink/monitor`、`/resource_status`、`/adaptive/status`、
`/migration_capacity?boundary=18`、`/admission`；
`POST /molink/reconfigure`、`/resize_kv`、`/admission`。
Admission Gate 使用有期限的租约，限制新请求进入而让已运行请求排空。

## 目录

- `vllm/distributed/pp_reconfigure/`：迁移、KV 管理、准入、监控、校准与控制命令。
- `vllm/distributed/pp_reconfigure/adaptive/`：成本模型、优化器、在线调度、TEE 接口。
- `vllm/entrypoints/serve/dev/molink/`：管理 API 与调度器生命周期。
- `vllm/v1/`、`vllm/model_executor/`：推理引擎和模型集成。
- `scripts/serve.sh`：统一启动入口。
- `requirements/`、`csrc/`、`cmake/`、`rust/`、`tools/`：依赖和上游构建支持。

运行数据写入被 Git 忽略的 `runtime/`；模型权重、日志和校准产物不属于发布源码。
许可证为 Apache-2.0，见 `LICENSE`。
