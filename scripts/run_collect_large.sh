#!/usr/bin/env bash
# Collection chain for the remaining items (2026-09-25):
#  1) weather-focused evaluation set: 4 new routes per town (seed offset 300) in ClearNoon, then the same
#     routes replayed under HardRainNoon / fog / ClearNight -> dataset_weather2/
#  2) large training set: 42 new routes per town (seed offset 30) in ClearNoon -> dataset/ (+252 episodes)
# Complete episodes are skipped on rerun, so the script can be restarted.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
LOG=artifacts_large/collect.log
CFG=configs/extended.yaml
TOWNS="Town01 Town02 Town03 Town04 Town05 Town10HD_Opt"
echo "=== $(date '+%F %T') WEATHER2 CLEAR START" >> $LOG
"$PY" -m rrf.collector --config $CFG --dataset-root dataset_weather2 --seed-offset 300 --episodes-per-town 4 >> $LOG 2>&1
IDS=$(ls -d dataset_weather2/*_ep*_seed* | xargs -n1 basename | grep -v "__")
for w in HardRainNoon fog ClearNight; do
  echo "=== $(date '+%F %T') WEATHER2 $w START" >> $LOG
  "$PY" -m rrf.collector --config $CFG --dataset-root dataset_weather2 --weather $w --tag __$w --replay $IDS >> $LOG 2>&1
done
echo "=== $(date '+%F %T') WEATHER2 DONE" >> $LOG
echo "=== $(date '+%F %T') LARGE START" >> $LOG
"$PY" -m rrf.collector --config $CFG --seed-offset 30 --episodes-per-town 42 >> $LOG 2>&1
echo "=== $(date '+%F %T') LARGE DONE" >> $LOG
