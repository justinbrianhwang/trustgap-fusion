#!/usr/bin/env bash
# Runs required by pre-registration revision 2 (B3A x 5 seeds, PG seeds 3-4, both models), with the same
# rerun-until-final.json policy as before, then the approval-gated test-v2 evaluation.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="${RRF_PYTHON:-python}"; PYTF="${RRF_TF_PYTHON:-python}"
LOG=artifacts_journal/pipeline.log
mark() { echo "=== $(date '+%F %T') $*" >> $LOG; }
if_run() { local m=$1 s=$2; for a in 1 2 3; do [ -f artifacts_journal/runs/fusion_${m}_seed$s/final.json ] && return
  "$PY" -m rrf.train --config configs/journal.yaml --stage fusion --method $m --seed $s > artifacts_journal/train_logs/if_fus_${m}_s${s}_a$a.log 2>&1; done; }
tf_run() { local m=$1 s=$2; for a in 1 2 3; do [ -f artifacts_journal/tf_runs/transfuser/fusion_${m}_seed$s/final.json ] && return
  "$PYTF" -m rrf.tf_train --config configs/journal.yaml --variant transfuser --stage fusion --method $m --seed $s > artifacts_journal/train_logs/tf_fus_${m}_s${s}_a$a.log 2>&1; done; }
mark "REV2 TRAIN START"
( for s in 0 1 2; do if_run B3A $s & done; wait; for job in "B3A 3" "B3A 4" "PG 3"; do if_run $job & done; wait; if_run PG 4 ) &
( for s in 0 1; do tf_run B3A $s & done; wait; for s in 2 3; do tf_run B3A $s & done; wait; tf_run B3A 4 & tf_run PG 3 & wait; tf_run PG 4 ) &
wait
mark "REV2 TRAIN DONE"
bash scripts/run_journal_test.sh
mark "REV2 TEST DONE"
