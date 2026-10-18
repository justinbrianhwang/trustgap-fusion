#!/usr/bin/env bash
# After the reliability stage: rerun (one at a time) any TransFuser estimator that is missing, e.g. after a
# transient GPU out-of-memory kill, then start TransFuser fusion. Also reruns the B3 diagnostics after the
# InterFuser diagnostics queue is done.
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PYTF="C:/Users/sunju/miniconda3/envs/rrf_tf/python.exe"
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
CK=artifacts_journal/tf_checkpoints/transfuser
until grep -q "RELIABILITY DONE" artifacts_journal/pipeline.log; do sleep 120; done
for s in 0 1 2 3 4; do for t in absolute relative; do
  [ $t = relative ] && [ $s -gt 2 ] && continue
  name=reliability_seed$s.pt; [ $t = relative ] && name=reliability_rel_seed$s.pt
  for a in 1 2 3; do
    grep -q "Traceback" artifacts_journal/train_logs/tf_rel_${t:0:3}_s$s.log 2>/dev/null || { [ -f $CK/$name ] && break; }
    echo "=== $(date '+%F %T') rerun TF reliability $t seed $s attempt $a" >> artifacts_journal/pipeline.log
    "$PYTF" -m rrf.tf_train --config configs/journal.yaml --variant transfuser --stage reliability --seed $s --target $t > artifacts_journal/train_logs/tf_rel_${t:0:3}_s$s.log 2>&1
  done
done; done
echo "=== $(date '+%F %T') TF RELIABILITY VERIFIED" >> artifacts_journal/pipeline.log
bash scripts/run_journal.sh tf_fusion &
until grep -q "DIAGNOSTICS DONE" artifacts_ext/diagnostics/run.log; do sleep 120; done
echo "=== $(date '+%F %T') START --method B3 --seed 0 (rerun after OOM)" >> artifacts_ext/diagnostics/run.log
"$PY" -m rrf.tools.diagnose --config configs/extended.yaml --split test --suite reduced --frame-stride 2 --method B3 --seed 0 --analyses shuffle,trace >> artifacts_ext/diagnostics/run.log 2>&1
echo "=== $(date '+%F %T') EXIT $? --method B3 rerun; B3 DIAG DONE" >> artifacts_ext/diagnostics/run.log
wait
