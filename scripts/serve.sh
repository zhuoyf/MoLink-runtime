#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${1:?Usage: bash scripts/serve.sh MODEL [vllm serve options]}"
shift
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_BATCH_INVARIANT=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SERVER_DEV_MODE=1 VLLM_PP_RECONFIGURE_ENABLED=1
export VLLM_PP_RECONFIGURE_ONLINE=1 VLLM_PP_RECONFIGURE_KV_TRANSPORT=pp
export VLLM_PP_RECONFIGURE_STACK_SIZE="${VLLM_PP_RECONFIGURE_STACK_SIZE:-2}"
export HETVLLM_PP_LINK=0 HETVLLM_RESOURCE_MONITOR=1
export VLLM_PP_RECONFIGURE_LOG_DIR="${VLLM_PP_RECONFIGURE_LOG_DIR:-runtime/migrations}"
export HETVLLM_RESOURCE_MONITOR_LOG="${HETVLLM_RESOURCE_MONITOR_LOG:-runtime/resources}"
exec .venv/bin/vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --distributed-executor-backend mp --dtype bfloat16 \
  --enforce-eager --attention-backend TRITON_ATTN \
  --generation-config vllm "$@"
