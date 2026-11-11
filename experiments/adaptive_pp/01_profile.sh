#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
if ! .venv/bin/python experiments/adaptive_pp/network_profile.py --output "$OUT/network.json" --check; then
  .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
    experiments/adaptive_pp/network_profile.py --output "$OUT/network.json"
fi
HISTORY="${HISTORY:-/workspace/molink-reconfiguration-results/pipelive/research-20261106/results/generation-lock-performance.json}"
if [[ -f "$HISTORY" && ! -f "$OUT/historical_profile.json" ]]; then
  .venv/bin/python experiments/adaptive_pp/import_history.py "$HISTORY" "$OUT/historical_profile.json"
fi
ensure_server
run_experiment profile "$@"
