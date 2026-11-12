#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
DIR="$PWD/experiments/concurrency_benchmark"
mkdir -p "$DIR/logs"
exec 9>"$DIR/run.lock"
flock -n 9 || exit 1
[[ ! -e "$DIR/results" ]] || { echo 'Results already exist'; exit 1; }
echo $$ > "$DIR/run.pid"
finish() {
  code=$?
  trap - EXIT
  if (( code != 0 )); then
    .venv/bin/python "$DIR/report.py" > "$DIR/logs/partial-report.log" 2>&1 || true
  fi
  echo "$code" > "$DIR/exit_code"
  date -u +%FT%TZ > "$DIR/completed"
  echo "END exit_code=$code"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
sha256sum -c "$DIR/frozen.sha256"
test -f "$DIR/pilot/PASSED"
mkdir "$DIR/results"
labels=(full threshold static_4 static_8 static_12 static_16 static_20 static_24 static_28)
for repeat in 0 1 2; do
  for concurrency in 2 4 8 12 16 24 32 48 64; do
    for index in $(seq 0 8); do
      offset=$(((index+repeat)%9))
      if (( repeat == 1 )); then offset=$((8-offset)); fi
      label=${labels[$offset]}
      name="r${repeat}_c${concurrency}_${label}"
      echo "START $name $(date -u +%FT%TZ)"
      bash "$DIR/run_trial.sh" "$label" "$concurrency" 200 "$DIR/results/$name"
      echo "DONE $name $(date -u +%FT%TZ)"
    done
  done
done
.venv/bin/python "$DIR/report.py"
