#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
.venv/bin/python -m pytest tests/v1/worker/test_adaptive_pp.py \
  tests/v1/worker/test_pp_reconfigure.py \
  -k 'admission_recovery or admission_lease or async_streams_before_pause' -q
run_case failure "$@"
