#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export MOLINK_HETEROGENEITY_SIMULATION=1
export MODEL=/workspace/model/llama7b GPU_MEMORY_UTILIZATION=0.80
export VLLM_PP_LAYER_PARTITION=16,16 VLLM_PP_RECONFIGURE_STACK_SIZE=4
export LD_PRELOAD=/workspace/molink-reconfiguration-results/pipelive/environment/nccl_shm_redirect.so
export VLLM_PP_RECONFIGURE_LOG_DIR="$PWD/experiments/heterogeneous_benchmark/logs/migrations"
export HETVLLM_RESOURCE_MONITOR_LOG="$PWD/experiments/heterogeneous_benchmark/logs/resources"
unset VLLM_PP_ADAPTIVE_PROFILE
exec bash experiments/adaptive_pp/serve.sh --no-enable-prefix-caching --max-num-seqs 4
