#!/usr/bin/env bash
# Closed loop with the EXTENDED checkpoints on the six searched routes of closed_loop_v2 (signals ignored,
# same as v2). One process per method with retries: the CARLA client occasionally aborts; finished runs are
# skipped on retry. Usage: bash scripts/run_closed_loop_ext.sh METHOD...
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="${RRF_PYTHON:-python}"
ROUTES="Town01_ep02_seed3702 Town02_ep00_seed3800 Town02_ep01_seed3801 Town03_ep00_seed3900 Town05_ep02_seed4102 Town10HD_Opt_ep02_seed9402"
mkdir -p artifacts_ext/eval_logs
# one CARLA client at a time (the follow-up orchestrator may call this script while another call is running)
LOCK=artifacts_ext/closed_loop.lock
while ! mkdir "$LOCK" 2>/dev/null; do sleep 60; done
trap 'rmdir "$LOCK"' EXIT
for m in "$@"; do
  for try in 1 2 3 4; do
    echo "=== $(date '+%F %T') closed loop $m try $try" >> artifacts_ext/eval_logs/closed_loop_$m.log
    "$PY" -m rrf.closed_loop --config configs/extended.yaml --methods $m --routes $ROUTES --ignore-signals --max-ticks 2400 --out artifacts_ext/closed_loop >> artifacts_ext/eval_logs/closed_loop_$m.log 2>&1 && break
  done
done
echo "=== CLOSED LOOP DONE ($*) $(date '+%F %T')" >> artifacts_ext/pipeline.log
