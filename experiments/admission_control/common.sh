#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
run_case() {
  .venv/bin/python experiments/admission_control/run.py "$1" "${@:2}"
}
