#!/bin/bash
# Usage: bash run_method_e.sh <task> <model> <hf_model> <data_name> [alpha] [base] [training flags...]
# HF model and alpha/base are retained for launcher compatibility.
# Pass --n-samples N in training flags to set trajectories per problem (default: 8).
set -euo pipefail

TASK="$1"
MODEL="$2"
HF_MODEL="$3"
DATA_NAME="$4"
ALPHA="${5:-0.5}"
BASE="${6:-0.1}"
RUN_ID="${MODEL}"

if [[ "${TASK}" == "math" ]]; then
  BASE_MODEL_DATA_NAME="math_o1"
elif [[ "${TASK}" == "sql" ]]; then
  BASE_MODEL_DATA_NAME="sql_conceptual"
else
  BASE_MODEL_DATA_NAME="${DATA_NAME}"
fi

python -m pipeline train_rl \
  --task "${TASK}" \
  --method method_e \
  --run-id "${RUN_ID}" \
  --base-model "models/${BASE_MODEL_DATA_NAME}/method_ac_rl/${RUN_ID}/model" \
  --data-name "${DATA_NAME}" \
  --sft-stride 128 \
  --sft-micro-batch-size 8 \
  --test-freq 25 \
  --keep-checkpoints \
  "${@:7}"
