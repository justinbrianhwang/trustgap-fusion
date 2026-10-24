#!/usr/bin/env bash
# Pre-registered diagnostics (section 6) on the journal seed-0 checkpoints, test v2, reduced suite, every
# 2nd frame. One InterFuser lane and one TransFuser lane in parallel, three attempts per job.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="${RRF_PYTHON:-python}"
PYTF="${RRF_TF_PYTHON:-python}"
LOG=artifacts_journal/diagnostics_run.log
ALL=utility,partial,oracle_rel,oracle_gate,oracle_gate_xfit,oracle_gate_temporal,shuffle,trace
COMMON=(--config configs/journal.yaml --split test --suite reduced --frame-stride 2 --seed 0)

job() {  # $1 = label, rest = command
  local label=$1; shift
  for attempt in 1 2 3; do
    echo "=== $(date '+%F %T') START $label attempt $attempt" >> $LOG
    "$@" >> "$LOG.$label" 2>&1
    local rc=$?
    echo "=== $(date '+%F %T') EXIT $rc $label" >> $LOG
    [ $rc -eq 0 ] && return 0
  done
}

if_lane() {
  for m in P PG B3A B4 PS PT PR B3 PGS PGT; do
    job "if_$m" "$PY" -m rrf.tools.diagnose "${COMMON[@]}" --method $m --analyses $ALL --batch-size 8
  done
  for m in B0 B2 BQ; do
    job "if_$m" "$PY" -m rrf.tools.diagnose "${COMMON[@]}" --method $m --analyses utility --batch-size 8
  done
}

tf_lane() {
  for m in P PG B3A B4 PS PT PR B3 PGS PGT; do
    job "tf_$m" "$PYTF" -m rrf.tf_diagnose "${COMMON[@]}" --method $m --analyses $ALL --batch-size 4
  done
  for m in B0 B2 BQ; do
    job "tf_$m" "$PYTF" -m rrf.tf_diagnose "${COMMON[@]}" --method $m --analyses utility --batch-size 4
  done
}

if_lane & tf_lane & wait
"$PY" -m rrf.tools.diagnostics_report artifacts_journal --out artifacts_journal/diagnostics_report >> $LOG 2>&1
echo "=== $(date '+%F %T') JOURNAL DIAGNOSTICS DONE" >> $LOG
