#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export VLLM_MQ_MAX_CHUNK_BYTES_MB=1
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SERVER_DEV_MODE=1
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-8,2}"
export VLLM_PP_RECONFIGURE_ENABLED=1
export VLLM_PP_RECONFIGURE_ONLINE=1
export VLLM_PP_RECONFIGURE_STACK_SIZE="${VLLM_PP_RECONFIGURE_STACK_SIZE:-2}"
export VLLM_PP_RECONFIGURE_KV_TRANSPORT=pp
export HETVLLM_PP_LINK=0
export HETVLLM_RESOURCE_MONITOR=0
export VLLM_PP_RECONFIGURE_LOG_DIR="${VLLM_PP_RECONFIGURE_LOG_DIR:-benchmark_results/pp_reconfigure/migrations}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
exec .venv/bin/vllm serve "${MODEL:?Set MODEL to a local checkpoint directory}" \
  --served-model-name "${SERVED_MODEL_NAME:-pp-reconfigure-test}" --host 127.0.0.1 --port 26020 \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --distributed-executor-backend mp --dtype bfloat16 --load-format safetensors \
  --max-model-len 512 --max-num-seqs 4 --max-num-batched-tokens 128 \
  --num-gpu-blocks-override 64 --gpu-memory-utilization 0.08 \
  --generation-config vllm --enforce-eager --attention-backend TRITON_ATTN "$@"
