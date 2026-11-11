#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
.venv/bin/python -m pytest tests/v1/worker/test_adaptive_pp.py -q -k security \
  --junitxml="$OUT/security.xml"
