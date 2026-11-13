#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export MOLINK_HETEROGENEITY_SIMULATION=1
export PORT="${PORT:-26030}" GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.06}"
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-6,4}"
export VLLM_PP_RECONFIGURE_LOG_DIR="$PWD/experiments/admission_control/results/migrations"
export HETVLLM_RESOURCE_MONITOR_LOG="$PWD/experiments/admission_control/results/resources"
export LD_PRELOAD="${LD_PRELOAD:-/workspace/molink-reconfiguration-results/pipelive/environment/nccl_shm_redirect.so}"
unset VLLM_PP_ADAPTIVE_PROFILE
exec bash experiments/adaptive_pp/serve.sh --max-num-seqs 64 \
  --num-gpu-blocks-override 192 --no-enable-prefix-caching "$@"
