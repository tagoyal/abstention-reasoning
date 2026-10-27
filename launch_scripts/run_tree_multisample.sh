#!/bin/bash
# Usage: bash launch_scripts/run_tree_multisample.sh <task> <run_id> <data_name> \
#   [gpus=1] [stride=128] [num_points=5] [num_samples=8] [split=rl_train]

set -euo pipefail

TASK="$1"
RUN_ID="$2"
DATA_NAME="$3"
GPUS="${4:-1}"
STRIDE="${5:-128}"
NUM_POINTS="${6:-5}"
NUM_SAMPLES="${7:-8}"
SPLIT="${8:-rl_train}"
MODEL_PATH="models/${DATA_NAME}/baseline_rl/${RUN_ID}/model"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Baseline RL model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

python -m pipeline generate_tree_multisample \
  --task "${TASK}" \
  --data-name "${DATA_NAME}" \
  --method baseline \
  --model "${MODEL_PATH}" \
  --run-id "${RUN_ID}" \
  --split "${SPLIT}" \
  --stride "${STRIDE}" \
  --num-points "${NUM_POINTS}" \
  --num-samples "${NUM_SAMPLES}" \
  --data-parallel-size "${GPUS}" \
  --async
