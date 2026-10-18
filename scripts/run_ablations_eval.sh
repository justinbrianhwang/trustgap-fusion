#!/usr/bin/env bash
# Evaluate ablations (B1/B3/B4 seed 0) and extra seeds (B2/P seeds 1,2) on the locked test suite, then summarize.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
LOG=artifacts/pilot_ablations.log
run() { echo "=== $(date '+%F %T') START: $*" >> $LOG; "$PY" "$@" >> $LOG 2>&1; echo "=== $(date '+%F %T') EXIT $?: $*" >> $LOG; }
for m in B1 B3 B4; do [ -f artifacts/checkpoints/fusion_${m}_seed0.pt ] && run -m rrf.tools.evaluate --method $m --suite full --split test; done
for s in 1 2; do for m in B2 P; do [ -f artifacts/checkpoints/fusion_${m}_seed$s.pt ] && run -m rrf.tools.evaluate --method $m --seed $s --suite full --split test; done; done
run -m rrf.tools.summarize --suite full --split test
for s in 1 2; do [ -f artifacts/metrics/P_seed${s}_full_test/frames.jsonl ] && run -m rrf.tools.reliability_report --method P --seed $s; done
run -m rrf.tools.report
echo "=== ABLATIONS DONE $(date '+%F %T')" >> $LOG
