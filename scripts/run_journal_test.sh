#!/usr/bin/env bash
# Approved-only test-v2 evaluation: two workers, three attempts per job, then common AP and inference.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY="${RRF_PYTHON:-python}"
PYTF="${RRF_TF_PYTHON:-python}"
DOC=docs/PREREGISTRATION_JOURNAL.md

if ! grep -Eq '^Approved:.*[0-9]{4}-[0-9]{2}-[0-9]{2}' "$DOC"; then
  echo "Refusing test-v2 evaluation: $DOC has no Approved: line containing a date." >&2
  exit 2
fi

mkdir -p artifacts_journal/test_logs artifacts_journal_weak/test_logs
status=0
pids=()

run_retry() {
  local log=$1; shift
  local attempt
  for attempt in 1 2 3; do
    echo "=== $(date '+%F %T') attempt $attempt: $*" >> "$log"
    "$@" >> "$log" 2>&1 && return 0
  done
  echo "FAILED after 3 attempts: $*" >&2
  return 1
}

wait_one() {
  local pid=${pids[0]}
  wait "$pid" || status=1
  pids=("${pids[@]:1}")
}

launch() {
  run_retry "$@" &
  pids+=("$!")
  [ "${#pids[@]}" -lt 2 ] || wait_one
}

drain() {
  while [ "${#pids[@]}" -gt 0 ]; do wait_one; done
  [ "$status" -eq 0 ] || { echo "At least one job failed; stopping." >&2; exit 1; }
}

queue_interfuser() {
  local root=$1 config=$2 final base stem method seed
  for final in "$root"/runs/fusion_*_seed*/final.json; do
    [ -f "$final" ] || continue
    base=$(basename "$(dirname "$final")")
    stem=${base#fusion_}; method=${stem%_seed*}; seed=${stem##*_seed}
    launch "$root/test_logs/if_${method}_seed${seed}.log" "$PY" -m rrf.tools.evaluate \
      --config "$config" --method "$method" --seed "$seed" --suite full --split test --frame-stride 2
  done
}

queue_transfuser() {
  local root=$1 config=$2 final base stem method seed stages extra
  for final in "$root"/tf_runs/transfuser/fusion_*_seed*/final.json; do
    [ -f "$final" ] || continue
    base=$(basename "$(dirname "$final")")
    stem=${base#fusion_}; seed=${stem##*_seed}; method=${stem%_seed*}; extra=()
    if [[ "$method" =~ ^(.+)_st([0-9]+)$ ]]; then
      method=${BASH_REMATCH[1]}; stages=${BASH_REMATCH[2]}; extra=(--stages "$stages")
    fi
    launch "$root/test_logs/tf_${base}.log" "$PYTF" -m rrf.tf_evaluate \
      --config "$config" --artifacts-root "$root" --variant transfuser --method "$method" \
      --root dataset_v2 --seed "$seed" --suite full --split test --frame-stride 2 "${extra[@]}"
  done
}

# Released baselines, including the two requested TransFuser fusion structures.
launch artifacts_journal/test_logs/if_B0.log "$PY" -m rrf.tools.evaluate \
  --config configs/journal.yaml --method B0 --suite full --split test --frame-stride 2
for variant in transfuser geometric_fusion; do
  launch "artifacts_journal/test_logs/tf_${variant}_B0.log" "$PYTF" -m rrf.tf_evaluate \
    --config configs/journal.yaml --artifacts-root artifacts_journal --variant "$variant" \
    --root dataset_v2 --method B0 --suite full --split test --frame-stride 2
done
queue_interfuser artifacts_journal configs/journal.yaml
queue_interfuser artifacts_journal_weak configs/journal_weak.yaml
queue_transfuser artifacts_journal configs/journal.yaml
queue_transfuser artifacts_journal_weak configs/journal_weak.yaml
drain

# Common-region AP for every completed prediction directory, still capped at two workers.
for root_config in "artifacts_journal configs/journal.yaml" "artifacts_journal_weak configs/journal_weak.yaml"; do
  set -- $root_config; root=$1; config=$2
  for directory in "$root"/metrics/*_full_test; do
    [ -f "$directory/pred_clean.npz" ] || continue
    name=$(basename "$directory")
    launch "$root/test_logs/common_if_${name}.log" "$PY" -m rrf.common_ap \
      --config "$config" --pred-dir "$directory" --model interfuser
  done
  for directory in "$root"/tf_metrics/*_full_test; do
    [ -f "$directory/pred_clean.npz" ] || continue
    name=$(basename "$directory")
    launch "$root/test_logs/common_tf_${name}.log" "$PY" -m rrf.common_ap \
      --config "$config" --pred-dir "$directory" --model transfuser
  done
done
drain

run_retry artifacts_journal/test_logs/journal_summary.log "$PY" -m rrf.tools.journal_summary \
  --config configs/journal.yaml --comparisons configs/journal_comparisons.json \
  --artifacts-root artifacts_journal --weak-artifacts-root artifacts_journal_weak || exit 1
echo "=== journal test evaluation complete $(date '+%F %T')"
