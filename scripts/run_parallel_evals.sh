#!/usr/bin/env bash
# Evaluate ablations and extra seeds on the locked test suite in parallel (independent processes).
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="${RRF_PYTHON:-python}"
mkdir -p artifacts/eval_logs
pids=()
for spec in "B1 0" "B3 0" "B4 0" "B2 1" "P 1" "B2 2" "P 2"; do
  set -- $spec
  "$PY" -m rrf.tools.evaluate --method $1 --seed $2 --suite full --split test > artifacts/eval_logs/${1}_seed$2.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== all parallel evaluations finished $(date '+%F %T')"
