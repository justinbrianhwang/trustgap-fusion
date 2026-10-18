#!/usr/bin/env bash
# First-pass diagnostics on the existing extension checkpoints (legacy 4-episode test, reduced suite,
# every 2nd frame). Gated methods: all analyses; B0/B2: utility only (no gate). Two processes at a time.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="C:/Users/sunju/miniconda3/envs/rrf/python.exe"
LOG=artifacts_ext/diagnostics/run.log
mkdir -p artifacts_ext/diagnostics
d() { echo "=== $(date '+%F %T') START $*" >> $LOG; "$PY" -m rrf.tools.diagnose --config configs/extended.yaml --split test --suite reduced --frame-stride 2 "$@" >> $LOG 2>&1; rc=$?; echo "=== $(date '+%F %T') EXIT $rc $*" >> $LOG; }
ALL=utility,oracle_rel,oracle_gate,shuffle,trace
(d --method P --seed 0 --analyses shuffle,trace; d --method PS --seed 0 --analyses $ALL; d --method PG --seed 0 --analyses $ALL; d --method B3 --seed 0 --analyses $ALL; d --method B0 --seed 0 --analyses utility) &
(d --method PT --seed 0 --analyses shuffle,trace; d --method PGS --seed 0 --analyses $ALL; d --method PR --seed 0 --analyses $ALL; d --method B4 --seed 0 --analyses $ALL; d --method B2 --seed 0 --analyses utility) &
wait
echo "=== $(date '+%F %T') DIAGNOSTICS DONE" >> $LOG
