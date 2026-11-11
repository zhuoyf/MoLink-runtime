#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
if [[ "${1:-}" != "--worker" ]]; then
  nohup bash "$ROOT/experiments/adaptive_pp/run_all.sh" --worker > "$OUT/run_all.log" 2>&1 &
  echo "$!" > "$OUT/run_all.pid"
  echo "Started PID $!; log: $OUT/run_all.log"
  exit 0
fi
for script in 01_profile 02_online_update 03_scheduler 04_comparison 05_security_interface; do
  bash "$ROOT/experiments/adaptive_pp/$script.sh" > "$OUT/$script.log" 2>&1
  echo "$script complete"
done
.venv/bin/python experiments/adaptive_pp/report.py --out "$OUT"
