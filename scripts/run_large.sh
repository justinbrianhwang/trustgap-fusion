#!/usr/bin/env bash
# Large-data experiment (configs/large.yaml, artifacts_large/): every valid episode outside the locked
# validation/test goes to train; budget 6000 steps. Usage: bash scripts/run_large.sh {splits|train|eval|report}
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
CFG=configs/large.yaml
LOG=artifacts_large/pipeline.log
mkdir -p artifacts_large/eval_logs
run() { echo "=== $(date '+%F %T') START: $*" >> $LOG; "$PY" "$@" >> $LOG 2>&1; echo "=== $(date '+%F %T') EXIT $?: $*" >> $LOG; }
runbg() { local name=$1; shift; "$PY" "$@" > artifacts_large/eval_logs/$name.log 2>&1 & }

case "${1:-all}" in
  splits)
    run -m rrf.tools.make_splits --config $CFG --no-image-hash --lock-from artifacts/manifests/split.json ;;
  train)
    for s in 0 1 2; do runbg rel_abs_s$s -m rrf.train --config $CFG --stage reliability --seed $s --target absolute; done; wait
    runbg rel_rel_s0 -m rrf.train --config $CFG --stage reliability --seed 0 --target relative; wait
    for s in 0 1 2; do runbg B2_s$s -m rrf.train --config $CFG --stage fusion --method B2 --seed $s; done; wait
    for s in 0 1 2; do runbg PS_s$s -m rrf.train --config $CFG --stage fusion --method PS --seed $s; done; wait
    for s in 0 1 2; do runbg P_s$s  -m rrf.train --config $CFG --stage fusion --method P  --seed $s; done; wait
    for s in 0 1 2; do runbg PT_s$s -m rrf.train --config $CFG --stage fusion --method PT --seed $s; done; wait
    for m in PG PGS PGT PR; do runbg ${m}_s0 -m rrf.train --config $CFG --stage fusion --method $m --seed 0; done; wait
    echo "=== TRAIN DONE $(date '+%F %T')" >> $LOG ;;
  eval)
    runbg eval_B0 -m rrf.tools.evaluate --config $CFG --method B0 --suite full --split test
    for s in 0 1 2; do for m in B2 PS; do runbg eval_${m}_s$s -m rrf.tools.evaluate --config $CFG --method $m --seed $s --suite full --split test; done; done; wait
    for s in 0 1 2; do for m in P PT; do runbg eval_${m}_s$s -m rrf.tools.evaluate --config $CFG --method $m --seed $s --suite full --split test; done; done; wait
    for m in PG PGS PGT PR; do runbg eval_${m}_s0 -m rrf.tools.evaluate --config $CFG --method $m --seed 0 --suite full --split test; done; wait
    for m in B0 B2 P PS PT PG PGS PGT PR; do runbg calib_$m -m rrf.tools.evaluate --config $CFG --method $m --suite clean --split validation --subgroup calibration_groups; done; wait
    echo "=== EVAL DONE $(date '+%F %T')" >> $LOG ;;
  report)
    run -m rrf.tools.summarize --config $CFG --suite full --split test
    run -m rrf.tools.operating_point --config $CFG --methods B0 B2 P PS PT PG PGS PGT PR
    for s in 0 1 2; do for m in P PS PT; do runbg rel_report_${m}_s$s -m rrf.tools.reliability_report --config $CFG --method $m --seed $s; done; done; wait
    run -m rrf.tools.object_ap --config $CFG
    run -m rrf.tools.evaluate_weather --config $CFG --methods B0 B2 P PS PT PG PGS PGT PR --seeds 0 1 2
    run -m rrf.tools.evaluate_weather --config $CFG --weather-root dataset_weather2 --clear-root dataset_weather2 --out-tag _w2 --methods B0 B2 P PS PT PG PGS PGT PR --seeds 0 1 2
    run -m rrf.tools.figures --config $CFG --methods B0 B2 P PS PT
    run -m rrf.tools.export_csv --config $CFG
    run -m rrf.tools.report --config $CFG --narrative docs/REPORT_NARRATIVE_LARGE.md
    echo "=== REPORT DONE $(date '+%F %T')" >> $LOG ;;
esac
