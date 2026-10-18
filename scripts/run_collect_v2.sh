#!/usr/bin/env bash
# Journal dataset v2 (dual rig: InterFuser + TransFuser sensors on the same drive), root dataset_v2/:
#   test v2  = 4 new routes per town (seed offset 500), val v2 = 2 per town (offset 600),
#   train    = replay of the 42 extension training routes (same town/seed/traffic seed).
# Complete episodes are skipped, so every command is repeated until its count is reached (CARLA client aborts).
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
CFG=configs/extended.yaml
ROOT=dataset_v2
LOG=dataset_v2_collect.log
TRAIN="Town01_ep00_seed3600 Town01_ep00_seed3610 Town01_ep00_seed3620 Town01_ep01_seed3621 Town01_ep02_seed3602 Town01_ep02_seed3622 Town01_ep03_seed3623 Town01_ep04_seed3624 Town02_ep00_seed3720 Town02_ep01_seed3701 Town02_ep01_seed3711 Town02_ep01_seed3721 Town02_ep02_seed3702 Town02_ep02_seed3722 Town02_ep03_seed3723 Town02_ep04_seed3724 Town03_ep00_seed3810 Town03_ep00_seed3820 Town03_ep01_seed3801 Town03_ep01_seed3821 Town03_ep02_seed3822 Town03_ep03_seed3823 Town03_ep04_seed3824 Town04_ep00_seed3920 Town04_ep01_seed3901 Town04_ep01_seed3921 Town04_ep02_seed3902 Town04_ep02_seed3922 Town04_ep03_seed3923 Town04_ep04_seed3924 Town05_ep00_seed4020 Town05_ep01_seed4001 Town05_ep01_seed4021 Town05_ep02_seed4022 Town05_ep03_seed4023 Town05_ep04_seed4024 Town10HD_Opt_ep00_seed9300 Town10HD_Opt_ep00_seed9320 Town10HD_Opt_ep01_seed9321 Town10HD_Opt_ep02_seed9322 Town10HD_Opt_ep03_seed9323 Town10HD_Opt_ep04_seed9324"
count() { "$PY" -c "import json,pathlib,sys; print(sum(1 for p in pathlib.Path('$ROOT').glob('*/episode.json') if json.loads(p.read_text()).get('complete')))"; }
for a in $(seq 1 15); do
  n=$(count); echo "=== $(date '+%F %T') attempt $a complete=$n/78" >> $LOG; [ "$n" -ge 78 ] && break
  "$PY" -m rrf.collector --config $CFG --rig both --dataset-root $ROOT --seed-offset 500 --episodes-per-town 4 >> $LOG 2>&1
  "$PY" -m rrf.collector --config $CFG --rig both --dataset-root $ROOT --seed-offset 600 --episodes-per-town 2 >> $LOG 2>&1
  "$PY" -m rrf.collector --config $CFG --rig both --dataset-root $ROOT --replay $TRAIN >> $LOG 2>&1
done
echo "=== $(date '+%F %T') COLLECT V2 DONE complete=$(count)/78" >> $LOG
