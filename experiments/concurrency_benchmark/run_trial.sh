#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
label=$1
concurrency=$2
seconds=$3
out=$4
boundary=16
[[ "$label" != static_* ]] || boundary=${label#static_}
[[ ! -e "$out" ]] || { echo "Output exists: $out"; exit 1; }
mkdir -p "$out-server"
if curl -fsS http://127.0.0.1:26020/health >/dev/null 2>&1; then
  echo 'Port occupied'; exit 1
fi
setsid bash experiments/concurrency_benchmark/serve.sh "$boundary" "$out-server" > "$out-server/server.log" 2>&1 &
server=$!
cleanup() {
  code=$?
  trap - EXIT
  kill -TERM -- "-$server" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 -- "-$server" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$server" 2>/dev/null || true
  wait "$server" 2>/dev/null || true
  exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
for _ in $(seq 1 240); do
  kill -0 "$server"
  if curl -fsS http://127.0.0.1:26020/health >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:26020/health >/dev/null
.venv/bin/python experiments/concurrency_benchmark/benchmark.py --label "$label" --concurrency "$concurrency" --seconds "$seconds" --out "$out"
