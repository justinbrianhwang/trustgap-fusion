#!/usr/bin/env bash
# Starts InterFuser fusion as soon as InterFuser's reliability estimators are done (TransFuser's may still be
# training), and TransFuser fusion after the whole reliability stage.
cd "$(dirname "$0")/.."
irel_running() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'rrf.train ' -and \$_.CommandLine -match 'reliability' } | Measure-Object).Count" | tr -d '\r '; }
until [ -f artifacts_journal/checkpoints/reliability_rel_seed2.pt ] && [ "$(irel_running)" = "0" ]; do sleep 120; done
bash scripts/run_journal.sh if_fusion &
until grep -q "RELIABILITY DONE" artifacts_journal/pipeline.log; do sleep 120; done
bash scripts/run_journal.sh tf_fusion &
wait
