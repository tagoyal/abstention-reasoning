#!/bin/bash
# Submit one job per (task, data_name, run_id), using baseline_rl checkpoints.
# Usage: bash launch_scripts/launch_tree_multisample_jobs.sh
# Override defaults with e.g. STRIDE=256 NUM_POINTS=3 GPUS=2 bash ...
# Push run_tree_multisample.sh and the pipeline changes before submitting:
# the pod clones the repository during initialization.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TASK_DATA=(
  countdown:countdown
  math:math_o1
  sql:sql_conceptual
)
RUN_IDS=(qwen2.5-1.5b qwen2.5-3b qwen3-4b qwen3-8b)

GPUS="${GPUS:-1}"
STRIDE="${STRIDE:-128}"
NUM_POINTS="${NUM_POINTS:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
SPLIT="${SPLIT:-rl_train}"

for PAIR in "${TASK_DATA[@]}"; do
  TASK="${PAIR%%:*}"
  DATA_NAME="${PAIR#*:}"
  for RUN_ID in "${RUN_IDS[@]}"; do
    sed "s|__TASK__|${TASK}|; s|__DATA_NAME__|${DATA_NAME}|; s|__RUN_ID__|${RUN_ID}|; s|__GPUS__|${GPUS}|; s|__STRIDE__|${STRIDE}|; s|__NUM_POINTS__|${NUM_POINTS}|; s|__NUM_SAMPLES__|${NUM_SAMPLES}|; s|__SPLIT__|${SPLIT}|" \
      "${SCRIPT_DIR}/submit_tree_multisample_template.yaml" \
      | kubectl create -f -
  done
done
