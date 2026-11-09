#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export VLLM_MQ_MAX_CHUNK_BYTES_MB=1
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SERVER_DEV_MODE=1
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-8,2}"
export VLLM_PP_RECONFIGURE_ENABLED=1
export HETVLLM_RESOURCE_MONITOR=1
export HETVLLM_RESOURCE_MONITOR_INTERVAL="${HETVLLM_RESOURCE_MONITOR_INTERVAL:-1}"
export VLLM_PP_RECONFIGURE_KV_TRANSPORT="${VLLM_PP_RECONFIGURE_KV_TRANSPORT:-nixl}"
export VLLM_PP_RECONFIGURE_LOG_DIR="${VLLM_PP_RECONFIGURE_LOG_DIR:-benchmark_results/pp_reconfigure/migrations}"
export HETVLLM_PP_LINK=1
export HETVLLM_PP_LINK_BIND=127.0.0.1
export HETVLLM_PP_LINK_PEER=127.0.0.1
export HETVLLM_PP_LINK_PORT=28100
export HETVLLM_PP_LINK_CODEC=bf16
export HETVLLM_PP_MERGE=0
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NCCL_BUFFSIZE=1048576
export UCX_TLS=tcp,self,cuda_copy
export NCCL_DEBUG=INFO
exec .venv/bin/vllm serve "${MODEL:?Set MODEL to a local checkpoint directory}" \
  --served-model-name "${SERVED_MODEL_NAME:-pp-reconfigure-test}" --host 127.0.0.1 --port 26010 \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --distributed-executor-backend mp --dtype bfloat16 \
  --load-format safetensors --max-model-len 512 --max-num-seqs 4 \
  --max-num-batched-tokens 128 --num-gpu-blocks-override 128 \
  --gpu-memory-utilization 0.06 --generation-config vllm --enforce-eager \
  "$@"
