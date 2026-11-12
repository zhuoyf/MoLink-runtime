#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
DIR="$PWD/experiments/heterogeneous_benchmark"
mkdir -p "$DIR/logs"
exec 9>"$DIR/run.lock"
flock -n 9 || { echo 'An experiment is already running'; exit 1; }
echo "$$" > "$DIR/run.pid"
rm -f "$DIR/exit_code" "$DIR/completed"
server_pid=''
finish() {
  code=$?
  trap - EXIT
  if [[ -n "$server_pid" ]]; then kill -TERM "$server_pid" 2>/dev/null || true; fi
  if (( code != 0 )) && [[ -d "$DIR/results" ]]; then
    .venv/bin/python "$DIR/report.py" > "$DIR/logs/partial-report.log" 2>&1 || true
  fi
  echo "$code" > "$DIR/exit_code"
  if (( code == 0 )); then echo COMPLETED > "$DIR/status"; else echo FAILED > "$DIR/status"; fi
  date -u +%FT%TZ > "$DIR/completed"
  echo "END exit_code=$code $(date -u +%FT%TZ)"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo RUNNING > "$DIR/status"
echo "START pid=$$ $(date -u +%FT%TZ)"
.venv/bin/python "$DIR/freeze.py" --check
if curl -fsS http://127.0.0.1:26020/health >/dev/null 2>&1; then
  echo 'Port 26020 already serves a model; refusing to mix experiments'; exit 1
fi
bash "$DIR/serve.sh" > "$DIR/logs/formal-server.log" 2>&1 &
server_pid=$!
echo "$server_pid" > "$DIR/server.pid"
for _ in $(seq 1 180); do
  kill -0 "$server_pid"
  if curl -fsS http://127.0.0.1:26020/health >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:26020/health >/dev/null
# Preflight and phase profiles were completed before this formal launch.
for script in 03_comparison 04_report; do
  echo "BEGIN $script $(date -u +%FT%TZ)"
  bash "$DIR/$script.sh" > "$DIR/logs/$script.log" 2>&1
  echo "DONE $script $(date -u +%FT%TZ)"
done
