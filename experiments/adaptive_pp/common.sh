#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="${OUT:-$ROOT/benchmark_results/adaptive}"
URL="${URL:-http://127.0.0.1:26020}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# Reuse the container transport setup from the existing PipeLive experiments.
SHM_PRELOAD="${SHM_PRELOAD:-/workspace/molink-reconfiguration-results/pipelive/environment/nccl_shm_redirect.so}"
if [[ -f "$SHM_PRELOAD" && ":${LD_PRELOAD:-}:" != *":$SHM_PRELOAD:"* ]]; then
  export LD_PRELOAD="$SHM_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
fi
ensure_server() {
  if ! curl -fsS "$URL/health" >/dev/null; then
    nohup bash "$ROOT/experiments/adaptive_pp/serve.sh" > "$OUT/server.log" 2>&1 &
    echo "$!" > "$OUT/server.pid"
    for _ in $(seq 1 180); do
      if curl -fsS "$URL/health" >/dev/null 2>&1; then return; fi
      sleep 1
    done
    echo "Server startup failed; see $OUT/server.log" >&2
    return 1
  fi
}
run_experiment() {
  .venv/bin/python experiments/adaptive_pp/experiment.py "$1" \
    --url "$URL" --out "$OUT" --model "${MODEL:-/workspace/model/molink-tiny-llama}" \
    --stack "${VLLM_PP_RECONFIGURE_STACK_SIZE:-2}" --repeats "${REPEATS:-3}" "${@:2}"
}
