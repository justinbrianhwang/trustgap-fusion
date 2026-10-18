#!/usr/bin/env bash
# End-of-queue check for TransFuser fusion (same policy as run_journal_fixup.sh).
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PYTF="C:/Users/sunju/miniconda3/envs/rrf_tf/python.exe"
LOG=artifacts_journal/pipeline.log
name() { "$PYTF" -c "from rrf.tf_train import fusion_name; print(fusion_name('$1', $2, tuple(int(x) for x in '$3'.split(','))))" 2>/dev/null | tail -1; }
plan() {
  for s in 0 1 2 3 4; do for m in B2 P PS PT; do echo "$m $s 0,1,2,3 --keep-all-checkpoints"; done; for m in BQ B3; do echo "$m $s 0,1,2,3"; done; done
  for s in 0 1 2; do for m in B4 PG PGS PGT PR; do echo "$m $s 0,1,2,3"; done; echo "P $s 0"; echo "P $s 3"; done
}
until grep -q "TF FUSION DONE" $LOG; do sleep 300; done
plan | while read m s st extra; do
  n=$(name $m $s $st)
  for a in 1 2 3; do
    [ -f artifacts_journal/tf_runs/transfuser/$n/final.json ] && break
    echo "=== $(date '+%F %T') rerun TF fusion $n attempt $a" >> $LOG
    "$PYTF" -m rrf.tf_train --config configs/journal.yaml --variant transfuser --stage fusion --method $m --seed $s --stages $st $extra > artifacts_journal/train_logs/tf_${n}_rerun$a.log 2>&1
  done
done
for s in 0 1 2; do for m in B2 P PS PT; do n=$(name $m $s 0,1,2,3); for a in 1 2 3; do
  [ -f artifacts_journal_weak/tf_runs/transfuser/$n/final.json ] && break
  echo "=== $(date '+%F %T') rerun TF weak $n attempt $a" >> $LOG
  "$PYTF" -m rrf.tf_train --config configs/journal_weak.yaml --variant transfuser --stage fusion --method $m --seed $s > artifacts_journal/train_logs/tf_weak_${n}_rerun$a.log 2>&1
done; done; done
echo "=== $(date '+%F %T') TF FUSION VERIFIED" >> $LOG
