#!/usr/bin/env bash
# End-of-queue check: after each fusion queue finishes, rerun (one at a time, up to 3 attempts) every run of
# the pre-registered plan that has no final.json (e.g. killed by a transient GPU out-of-memory).
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="${RRF_PYTHON:-python}"; PYTF="${RRF_TF_PYTHON:-python}"
LOG=artifacts_journal/pipeline.log
plan() {  # prints: method seed extra-args
  for s in 0 1 2 3 4; do for m in B2 P PS PT; do echo "$m $s --keep-all-checkpoints"; done; for m in BQ B3; do echo "$m $s"; done; done
  for s in 0 1 2; do for m in B4 PG PGS PGT PR; do echo "$m $s"; done; done
}
until grep -q "IF FUSION DONE" $LOG; do sleep 300; done
plan | while read m s extra; do
  for a in 1 2 3; do
    [ -f artifacts_journal/runs/fusion_${m}_seed$s/final.json ] && break
    echo "=== $(date '+%F %T') rerun IF fusion $m seed $s attempt $a" >> $LOG
    "$PY" -m rrf.train --config configs/journal.yaml --stage fusion --method $m --seed $s $extra > artifacts_journal/train_logs/if_fus_${m}_s${s}_rerun$a.log 2>&1
  done
done
for s in 0 1 2; do for m in B2 P PS PT; do for a in 1 2 3; do
  [ -f artifacts_journal_weak/runs/fusion_${m}_seed$s/final.json ] && break
  echo "=== $(date '+%F %T') rerun IF weak $m seed $s attempt $a" >> $LOG
  "$PY" -m rrf.train --config configs/journal_weak.yaml --stage fusion --method $m --seed $s > artifacts_journal/train_logs/if_weak_${m}_s${s}_rerun$a.log 2>&1
done; done; done
echo "=== $(date '+%F %T') IF FUSION VERIFIED" >> $LOG
