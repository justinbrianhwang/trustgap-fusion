#!/usr/bin/env bash
# Extended experiment (configs/extended.yaml, artifacts_ext/): more training episodes, same locked
# validation/test; methods B2, P, PG (gate-only), PR (relative target) x 3 seeds, B3/B4 seed 0.
# Usage: bash scripts/run_extended.sh {splits|train|eval|report}
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
CFG=configs/extended.yaml
LOG=artifacts_ext/pipeline.log
mkdir -p artifacts_ext/eval_logs
run() { echo "=== $(date '+%F %T') START: $*" >> $LOG; "$PY" "$@" >> $LOG 2>&1; echo "=== $(date '+%F %T') EXIT $?: $*" >> $LOG; }
runbg() { local name=$1; shift; "$PY" "$@" > artifacts_ext/eval_logs/$name.log 2>&1 & }

case "${1:-all}" in
  splits)
    run -m rrf.tools.make_splits --config $CFG --no-image-hash --lock-from artifacts/manifests/split.json ;;
  train)
    # reliability estimators (absolute + relative), 3 seeds, 3 concurrent
    for s in 0 1 2; do runbg rel_abs_s$s -m rrf.train --config $CFG --stage reliability --seed $s --target absolute; done; wait
    for s in 0 1 2; do runbg rel_rel_s$s -m rrf.train --config $CFG --stage reliability --seed $s --target relative; done; wait
    # fusion runs, 3 concurrent
    for s in 0 1 2; do runbg B2_s$s -m rrf.train --config $CFG --stage fusion --method B2 --seed $s; done; wait
    for s in 0 1 2; do runbg P_s$s  -m rrf.train --config $CFG --stage fusion --method P  --seed $s; done; wait
    for s in 0 1 2; do runbg PG_s$s -m rrf.train --config $CFG --stage fusion --method PG --seed $s; done; wait
    for s in 0 1 2; do runbg PR_s$s -m rrf.train --config $CFG --stage fusion --method PR --seed $s; done; wait
    runbg B3_s0 -m rrf.train --config $CFG --stage fusion --method B3 --seed 0
    runbg B4_s0 -m rrf.train --config $CFG --stage fusion --method B4 --seed 0; wait
    echo "=== TRAIN DONE $(date '+%F %T')" >> $LOG ;;
  eval)
    runbg eval_B0 -m rrf.tools.evaluate --config $CFG --method B0 --suite full --split test
    for m in B3 B4; do runbg eval_${m}_s0 -m rrf.tools.evaluate --config $CFG --method $m --seed 0 --suite full --split test; done
    for s in 0 1 2; do for m in B2 P; do runbg eval_${m}_s$s -m rrf.tools.evaluate --config $CFG --method $m --seed $s --suite full --split test; done; done; wait
    for s in 0 1 2; do for m in PG PR; do runbg eval_${m}_s$s -m rrf.tools.evaluate --config $CFG --method $m --seed $s --suite full --split test; done; done; wait
    for m in B0 B2 B3 B4 P PG PR; do runbg calib_$m -m rrf.tools.evaluate --config $CFG --method $m --suite clean --split validation --subgroup calibration_groups; done; wait
    echo "=== EVAL DONE $(date '+%F %T')" >> $LOG ;;
  report)
    run -m rrf.tools.summarize --config $CFG --suite full --split test
    run -m rrf.tools.operating_point --config $CFG --methods B0 B2 B3 B4 P PG PR
    for s in 0 1 2; do for m in P PG PR; do runbg rel_report_${m}_s$s -m rrf.tools.reliability_report --config $CFG --method $m --seed $s; done; done; wait
    run -m rrf.tools.object_ap --config $CFG
    run -m rrf.tools.figures --config $CFG --methods B0 B2 P PG PR
    run -m rrf.tools.export_csv --config $CFG
    run -m rrf.tools.report --config $CFG --narrative docs/REPORT_NARRATIVE_EXTENDED.md
    echo "=== REPORT DONE $(date '+%F %T')" >> $LOG ;;
esac

# ---- follow-ups (2026-09-25): token-scaling gate (PS joint / PGS gate-only), weather track, relative-target reliability report
case "${1:-}" in
  scale_train)
    for s in 0 1 2; do runbg PS_s$s -m rrf.train --config $CFG --stage fusion --method PS --seed $s; done; wait
    for s in 0 1 2; do runbg PGS_s$s -m rrf.train --config $CFG --stage fusion --method PGS --seed $s; done; wait
    echo "=== SCALE TRAIN DONE $(date '+%F %T')" >> $LOG ;;
  scale_eval)
    for s in 0 1 2; do for m in PS PGS; do runbg eval_${m}_s$s -m rrf.tools.evaluate --config $CFG --method $m --seed $s --suite full --split test; done; done; wait
    for m in PS PGS; do runbg calib_$m -m rrf.tools.evaluate --config $CFG --method $m --suite clean --split validation --subgroup calibration_groups; done; wait
    for s in 0 1 2; do for m in PS PGS; do runbg rel_report_${m}_s$s -m rrf.tools.reliability_report --config $CFG --method $m --seed $s; done; done; wait
    echo "=== SCALE EVAL DONE $(date '+%F %T')" >> $LOG ;;
  weather)
    run -m rrf.tools.evaluate_weather --config $CFG --methods B0 B2 P PG PR PS PGS B3 B4 --seeds 0 1 2 ;;
  relrep)
    for s in 0 1 2; do runbg rel_report_rel_PR_s$s -m rrf.tools.reliability_report --config $CFG --method PR --seed $s --target relative; done; wait
    echo "=== RELREP DONE $(date '+%F %T')" >> $LOG ;;
  report2)
    run -m rrf.tools.summarize --config $CFG --suite full --split test
    run -m rrf.tools.operating_point --config $CFG --methods B0 B2 B3 B4 P PG PR PS PGS
    run -m rrf.tools.object_ap --config $CFG
    run -m rrf.tools.figures --config $CFG --methods B0 B2 P PG PR PS PGS
    run -m rrf.tools.export_csv --config $CFG
    run -m rrf.tools.report --config $CFG --narrative docs/REPORT_NARRATIVE_EXTENDED.md
    echo "=== REPORT2 DONE $(date '+%F %T')" >> $LOG ;;
esac

# ---- key-scaling gate (PT joint / PGT gate-only, learned per-modality exponent), 2026-09-25
case "${1:-}" in
  temp_train)
    for s in 0 1 2; do runbg PT_s$s -m rrf.train --config $CFG --stage fusion --method PT --seed $s; done; wait
    for s in 0 1 2; do runbg PGT_s$s -m rrf.train --config $CFG --stage fusion --method PGT --seed $s; done; wait
    echo "=== TEMP TRAIN DONE $(date '+%F %T')" >> $LOG ;;
  temp_eval)
    for s in 0 1 2; do for m in PT PGT; do runbg eval_${m}_s$s -m rrf.tools.evaluate --config $CFG --method $m --seed $s --suite full --split test; done; done; wait
    for m in PT PGT; do runbg calib_$m -m rrf.tools.evaluate --config $CFG --method $m --suite clean --split validation --subgroup calibration_groups; done; wait
    for s in 0 1 2; do for m in PT PGT; do runbg rel_report_${m}_s$s -m rrf.tools.reliability_report --config $CFG --method $m --seed $s; done; done; wait
    run -m rrf.tools.evaluate_weather --config $CFG --methods PT PGT --seeds 0 1 2 --out-tag _temp
    echo "=== TEMP EVAL DONE $(date '+%F %T')" >> $LOG ;;
  report3)
    run -m rrf.tools.summarize --config $CFG --suite full --split test
    run -m rrf.tools.operating_point --config $CFG --methods B0 B2 B3 B4 P PG PR PS PGS PT PGT
    run -m rrf.tools.object_ap --config $CFG
    run -m rrf.tools.figures --config $CFG --methods B0 B2 P PG PR PS PGS PT PGT
    run -m rrf.tools.export_csv --config $CFG
    run -m rrf.tools.report --config $CFG --narrative docs/REPORT_NARRATIVE_EXTENDED.md
    echo "=== REPORT3 DONE $(date '+%F %T')" >> $LOG ;;
esac
