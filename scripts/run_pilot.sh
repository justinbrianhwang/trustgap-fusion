#!/usr/bin/env bash
# Sequential pilot pipeline: probes -> reliability -> B2 -> P -> full test evaluation -> summary.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="${RRF_PYTHON:-python}"
LOG=artifacts/pilot_pipeline.log
run() { echo "=== $(date '+%F %T') START: $*" >> $LOG; "$PY" "$@" >> $LOG 2>&1; echo "=== $(date '+%F %T') EXIT $?: $*" >> $LOG; }
mkdir -p artifacts
run -m rrf.train --stage reference --seed 0
run -m rrf.train --stage reliability --seed 0
run -m rrf.train --stage fusion --method B2 --seed 0
run -m rrf.train --stage fusion --method P --seed 0
run -m rrf.tools.evaluate --method B0 --suite full --split test
run -m rrf.tools.evaluate --method B2 --suite full --split test
run -m rrf.tools.evaluate --method P --suite full --split test
run -m rrf.tools.summarize --suite full --split test
for m in B0 B2 P; do run -m rrf.tools.evaluate --method $m --suite clean --split validation --subgroup calibration_groups; done
run -m rrf.tools.operating_point --methods B0 B2 P
run -m rrf.tools.reliability_report --method P
run -m rrf.tools.figures --methods B0 B2 P
echo "=== PIPELINE DONE $(date '+%F %T')" >> $LOG
