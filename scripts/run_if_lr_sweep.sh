#!/usr/bin/env bash
# Validation-only learning-rate check for InterFuser B2 on the v2 split (all journal-lr 2e-5 runs fell below
# the released model on validation v2). No test evaluation.
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
for tag in lr1em5 lr5em6; do for s in 0 1; do
  "$PY" -m rrf.train --config configs/journal_$tag.yaml --stage fusion --method B2 --seed $s > artifacts_journal/train_logs/sweep_${tag}_B2_s$s.log 2>&1 &
done; done; wait
echo "=== $(date '+%F %T') LR SWEEP DONE" >> artifacts_journal/pipeline.log
