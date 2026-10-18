#!/usr/bin/env bash
# Journal study on dataset_v2 (configs/journal*.yaml -> artifacts_journal*/), both models.
# Usage: bash scripts/run_journal.sh {reliability|fusion_core|fusion_aux|fusion_weak|tf_stages}
# Test-v2 evaluation is NOT part of this script until docs/PREREGISTRATION_JOURNAL.md is approved.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="${RRF_PYTHON:-python}"
PYTF="${RRF_TF_PYTHON:-python}"
CFG=configs/journal.yaml
LOG=artifacts_journal/pipeline.log
mkdir -p artifacts_journal/train_logs
if_bg() { local n=$1; shift; "$PY" -m rrf.train --config $CFG "$@" > artifacts_journal/train_logs/if_$n.log 2>&1 & }
tf_bg() { local n=$1; shift; "$PYTF" -m rrf.tf_train --config $CFG --variant transfuser "$@" > artifacts_journal/train_logs/tf_$n.log 2>&1 & }
mark() { echo "=== $(date '+%F %T') $*" >> $LOG; }

case "${1:-}" in
  reliability)
    mark "RELIABILITY START"
    for s in 0 1 2 3 4; do if_bg rel_abs_s$s --stage reliability --seed $s --target absolute; [ $s -eq 2 ] && wait; done; wait
    for s in 0 1 2; do if_bg rel_rel_s$s --stage reliability --seed $s --target relative; done; wait
    for s in 0 1 2 3 4; do tf_bg rel_abs_s$s --stage reliability --seed $s --target absolute; [ $s -eq 2 ] && wait; done; wait
    for s in 0 1 2; do tf_bg rel_rel_s$s --stage reliability --seed $s --target relative; done; wait
    mark "RELIABILITY DONE" ;;
  if_fusion)
    # InterFuser fusion runs, 3 at a time, pre-registered order (docs/PREREGISTRATION_JOURNAL.md section 3)
    mark "IF FUSION START"
    q=()
    for s in 0 1 2 3 4; do for m in B2 P PS PT; do q+=("$m $s --keep-all-checkpoints"); done; for m in BQ B3; do q+=("$m $s"); done; done
    for s in 0 1 2; do for m in B4 PG PGS PGT PR; do q+=("$m $s"); done; done
    n=0; for job in "${q[@]}"; do set -- $job; m=$1; s=$2; shift 2
      if_bg fus_${m}_s$s --stage fusion --method $m --seed $s "$@"; n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait; done; wait
    # weak augmentation: own artifacts root; reuses the standard-profile reliability estimators (copied)
    mkdir -p artifacts_journal_weak/checkpoints artifacts_journal_weak/manifests
    cp artifacts_journal/manifests/split.json artifacts_journal_weak/manifests/
    cp artifacts_journal/checkpoints/reliability_seed[0-2].pt artifacts_journal_weak/checkpoints/
    n=0; for s in 0 1 2; do for m in B2 P PS PT; do
      "$PY" -m rrf.train --config configs/journal_weak.yaml --stage fusion --method $m --seed $s > artifacts_journal/train_logs/if_weak_${m}_s$s.log 2>&1 &
      n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait; done; done; wait
    mark "IF FUSION DONE" ;;
  tf_fusion)
    mark "TF FUSION START"
    q=()
    for s in 0 1 2 3 4; do for m in B2 P PS PT; do q+=("$m $s --keep-all-checkpoints"); done; for m in BQ B3; do q+=("$m $s"); done; done
    for s in 0 1 2; do for m in B4 PG PGS PGT PR; do q+=("$m $s"); done; q+=("P $s --stages 0" "P $s --stages 3"); done
    n=0; for job in "${q[@]}"; do set -- $job; m=$1; s=$2; shift 2; tag=$(echo "$*" | tr -dc '0-9a-z'); 
      tf_bg fus_${m}_s${s}_$tag --stage fusion --method $m --seed $s "$@"; n=$((n+1)); [ $((n % 2)) -eq 0 ] && wait; done; wait
    mkdir -p artifacts_journal_weak/tf_checkpoints/transfuser
    cp artifacts_journal/tf_checkpoints/transfuser/reliability_seed[0-2].pt artifacts_journal_weak/tf_checkpoints/transfuser/
    n=0; for s in 0 1 2; do for m in B2 P PS PT; do
      "$PYTF" -m rrf.tf_train --config configs/journal_weak.yaml --variant transfuser --stage fusion --method $m --seed $s > artifacts_journal/train_logs/tf_weak_${m}_s$s.log 2>&1 &
      n=$((n+1)); [ $((n % 2)) -eq 0 ] && wait; done; done; wait
    mark "TF FUSION DONE" ;;
esac
