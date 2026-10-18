#!/usr/bin/env bash
# Robust completion pass for scripts/run_collect_large.sh. The CARLA client occasionally aborts
# ("Assertion failed: _episode != nullptr") and kills the collector; complete episodes are skipped on
# rerun, so every command is simply repeated until its episode count is reached (or 15 attempts).
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
LOG=artifacts_large/collect.log
CFG=configs/extended.yaml
count() { "$PY" - "$1" <<'PYEOF'
import json, sys, pathlib
print(sum(1 for p in pathlib.Path(sys.argv[1]).glob("*/episode.json") if json.loads(p.read_text()).get("complete")))
PYEOF
}
running() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'rrf.collector' } | Measure-Object).Count"; }
until [ "$(running | tr -d '\r ')" = "0" ]; do sleep 30; done
IDS=$(ls -d dataset_weather2/*_ep*_seed* | xargs -n1 basename | grep -v "__")
for attempt in $(seq 1 15); do
  n=$(count dataset_weather2); [ "$n" -ge 96 ] && break
  echo "=== $(date '+%F %T') WEATHER2 COMPLETION attempt $attempt ($n/96)" >> $LOG
  "$PY" -m rrf.collector --config $CFG --dataset-root dataset_weather2 --seed-offset 300 --episodes-per-town 4 >> $LOG 2>&1
  for w in HardRainNoon fog ClearNight; do
    "$PY" -m rrf.collector --config $CFG --dataset-root dataset_weather2 --weather $w --tag __$w --replay $IDS >> $LOG 2>&1
  done
done
for attempt in $(seq 1 15); do
  n=$(count dataset); [ "$n" -ge 305 ] && break
  echo "=== $(date '+%F %T') LARGE COMPLETION attempt $attempt ($n/305)" >> $LOG
  "$PY" -m rrf.collector --config $CFG --seed-offset 30 --episodes-per-town 42 >> $LOG 2>&1
done
echo "=== $(date '+%F %T') COLLECT ALL DONE weather2=$(count dataset_weather2)/96 dataset=$(count dataset)/305" >> $LOG
