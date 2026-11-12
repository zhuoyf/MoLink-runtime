#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python experiments/heterogeneous_benchmark/benchmark.py pilot
.venv/bin/python experiments/heterogeneous_benchmark/freeze.py
