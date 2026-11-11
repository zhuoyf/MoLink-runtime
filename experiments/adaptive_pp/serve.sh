#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SERVER_DEV_MODE=1 VLLM_PP_RECONFIGURE_ENABLED=1 VLLM_PP_RECONFIGURE_ONLINE=1
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-8,2}"
export VLLM_PP_RECONFIGURE_STACK_SIZE="${VLLM_PP_RECONFIGURE_STACK_SIZE:-2}"
export VLLM_PP_RECONFIGURE_KV_TRANSPORT=pp HETVLLM_PP_LINK=0 HETVLLM_RESOURCE_MONITOR=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_MQ_MAX_CHUNK_BYTES_MB=1
export VLLM_PP_RECONFIGURE_LOG_DIR="${VLLM_PP_RECONFIGURE_LOG_DIR:-benchmark_results/adaptive/migrations}"
export HETVLLM_RESOURCE_MONITOR_LOG="${HETVLLM_RESOURCE_MONITOR_LOG:-benchmark_results/adaptive/resources}"
exec .venv/bin/vllm serve "${MODEL:-/workspace/model/molink-tiny-llama}" \
  --served-model-name adaptive-test --host 127.0.0.1 --port "${PORT:-26020}" \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 --distributed-executor-backend mp \
  --dtype bfloat16 --load-format safetensors --max-model-len 512 --max-num-seqs 8 \
  --max-num-batched-tokens 128 --num-gpu-blocks-override 128 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.15}" \
  --generation-config vllm --enforce-eager --attention-backend TRITON_ATTN "$@"
