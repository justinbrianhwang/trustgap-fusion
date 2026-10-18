#!/usr/bin/env bash
# Cross-fitted gate-oracle control on the extension seed-0 checkpoints (legacy test), one process at a time.
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
LOG=artifacts_ext/diagnostics/run_xfit.log
for m in P PS PR PT B4 PG PGS B3; do
  for a in 1 2 3; do
    echo "=== $(date '+%F %T') START $m attempt $a" >> $LOG
    "$PY" -m rrf.tools.diagnose --config configs/extended.yaml --split test --suite reduced --frame-stride 2 --method $m --seed 0 --analyses oracle_gate_xfit --batch-size 4 >> $LOG 2>&1
    rc=$?; echo "=== $(date '+%F %T') EXIT $rc $m" >> $LOG; [ $rc -eq 0 ] && break
  done
done
"$PY" -m rrf.tools.diagnostics_report artifacts_ext --out artifacts_ext/diagnostics_report >> $LOG 2>&1
echo "=== $(date '+%F %T') XFIT DONE" >> $LOG
