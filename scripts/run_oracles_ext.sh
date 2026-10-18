#!/usr/bin/env bash
# One consistent pass (final code) of all gate-oracle variants on the extension seed-0 checkpoints (legacy
# test): in-sample, cross-fitted and temporal. Starts after the InterFuser fusion queue to keep GPU load low.
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
LOG=artifacts_ext/diagnostics/run_oracles.log
until grep -q "IF FUSION VERIFIED" artifacts_journal/pipeline.log; do sleep 300; done
for m in P PS PR PT B4 PG PGS PGT B3; do
  for a in 1 2 3; do
    echo "=== $(date '+%F %T') START $m attempt $a" >> $LOG
    "$PY" -m rrf.tools.diagnose --config configs/extended.yaml --split test --suite reduced --frame-stride 2 --method $m --seed 0 --analyses oracle_gate,oracle_gate_xfit,oracle_gate_temporal --batch-size 8 >> $LOG 2>&1
    rc=$?; echo "=== $(date '+%F %T') EXIT $rc $m" >> $LOG; [ $rc -eq 0 ] && break
  done
done
"$PY" -m rrf.tools.diagnostics_report artifacts_ext --out artifacts_ext/diagnostics_report >> $LOG 2>&1
echo "=== $(date '+%F %T') ORACLES DONE" >> $LOG
