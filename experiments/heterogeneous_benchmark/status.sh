#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f exit_code ]]; then
  echo "$(cat status) exit_code=$(cat exit_code) finished=$(cat completed)"
elif [[ -f run.pid ]] && kill -0 "$(cat run.pid)" 2>/dev/null; then
  echo "RUNNING pid=$(cat run.pid)"
else
  echo 'NOT RUNNING / ABNORMAL (no terminal exit_code)'
fi
