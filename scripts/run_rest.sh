#!/usr/bin/env bash
# Remainder of the pilot after B2 and the reliability estimator exist: P -> test evals -> reports.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="${RRF_PYTHON:-python}"
LOG=artifacts/pilot_rest.log
run() { echo "=== $(date '+%F %T') START: $*" >> $LOG; "$PY" "$@" >> $LOG 2>&1; echo "=== $(date '+%F %T') EXIT $?: $*" >> $LOG; }
run -m rrf.train --stage fusion --method P --seed 0
for m in B0 B2 P; do run -m rrf.tools.evaluate --method $m --suite full --split test; done
run -m rrf.tools.summarize --suite full --split test
for m in B0 B2 P; do run -m rrf.tools.evaluate --method $m --suite clean --split validation --subgroup calibration_groups; done
run -m rrf.tools.operating_point --methods B0 B2 P
run -m rrf.tools.reliability_report --method P
run -m rrf.tools.figures --methods B0 B2 P
run -m rrf.tools.report
echo "=== REST DONE $(date '+%F %T')" >> $LOG
