#!/usr/bin/env bash
# Weather-focused evaluation on dataset_weather2 (24 new routes x ClearNoon/HardRainNoon/fog/ClearNight),
# every 4th frame (2.5 Hz), ClearNoon counterpart evaluated once per checkpoint.
# Extended checkpoints: all seeds for the methods with a weather lead (B2, P, PG, PGS, PS), seed 0 otherwise;
# large checkpoints: seed 0 for every method.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="${RRF_PYTHON:-python}"
LOG=artifacts_ext/pipeline.log
run() { echo "=== $(date '+%F %T') START: $*" >> $LOG; "$PY" "$@" >> $LOG 2>&1; echo "=== $(date '+%F %T') EXIT $?: $*" >> $LOG; }
W2="--weather-root dataset_weather2 --clear-root dataset_weather2 --out-tag _w2 --frame-stride 4"
run -m rrf.tools.evaluate_weather --config configs/extended.yaml $W2 --methods B0 B2 P PG PGS PS --seeds 0 1 2
run -m rrf.tools.evaluate_weather --config configs/extended.yaml $W2 --out-tag _w2b --methods PR PT PGT B3 B4 --seeds 0
run -m rrf.tools.evaluate_weather --config configs/large.yaml $W2 --methods B0 B2 P PS PT PG PGS PGT PR --seeds 0
echo "=== WEATHER2 EVAL DONE $(date '+%F %T')" >> $LOG
