#!/usr/bin/env bash
# Budget ablation for the large-data experiment: B2 and PS x 3 seeds with the extension budget (3000
# steps) on the large training set; same split and reliability checkpoints as artifacts_large (copied).
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
CFG=configs/large_3k.yaml
LOG=artifacts_large_3k/pipeline.log
run() { echo "=== $(date '+%F %T') START: $*" >> $LOG; "$PY" "$@" >> $LOG 2>&1; echo "=== $(date '+%F %T') EXIT $?: $*" >> $LOG; }
runbg() { local name=$1; shift; "$PY" "$@" > artifacts_large_3k/eval_logs/$name.log 2>&1 & }
for s in 0 1 2; do runbg B2_s$s -m rrf.train --config $CFG --stage fusion --method B2 --seed $s; done; wait
for s in 0 1 2; do runbg PS_s$s -m rrf.train --config $CFG --stage fusion --method PS --seed $s; done; wait
echo "=== TRAIN DONE $(date '+%F %T')" >> $LOG
runbg eval_B0 -m rrf.tools.evaluate --config $CFG --method B0 --suite full --split test
for s in 0 1 2; do for m in B2 PS; do runbg eval_${m}_s$s -m rrf.tools.evaluate --config $CFG --method $m --seed $s --suite full --split test; done; done; wait
run -m rrf.tools.summarize --config $CFG --suite full --split test
run -m rrf.tools.export_csv --config $CFG
echo "=== ALL DONE $(date '+%F %T')" >> $LOG
