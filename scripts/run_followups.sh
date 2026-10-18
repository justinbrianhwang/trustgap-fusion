#!/usr/bin/env bash
# Follow-ups after the extension: token-scaling gate, relative-target reliability report, weather track,
# closed loop with extended checkpoints, then the report. Usage: bash scripts/run_followups.sh
set -u
cd "$(dirname "$0")/.."
bash scripts/run_closed_loop_ext.sh B0 B2 P PG PR &
CL=$!
bash scripts/run_extended.sh scale_train
bash scripts/run_extended.sh relrep &
R1=$!
bash scripts/run_extended.sh weather &
R2=$!
wait $R1 $R2
bash scripts/run_extended.sh scale_eval
wait $CL
bash scripts/run_closed_loop_ext.sh PS PGS
bash scripts/run_extended.sh report2
echo "=== FOLLOWUPS DONE $(date '+%F %T')" >> artifacts_ext/pipeline.log
