#!/bin/bash
# Usage: bash run_method_d.sh <task> <model> <hf_model> <data_name> [alpha] [base] [training flags...]
# HF model and alpha/base are accepted but unused for launcher compatibility.
# SFT only by default; pass --interleave-rl after alpha/base to alternate with RL.

set -euo pipefail

TASK="$1"
MODEL="$2"
HF_MODEL="$3"
DATA_NAME="$4"
ALPHA="${5:-0.5}"
BASE="${6:-0.1}"
RUN_ID="${MODEL}"

# The baseline_rl model used to seed method_d lives under a fixed
# data/model namespace per task family, regardless of which DATA_NAME
# variant is being trained on: math_o2 reuses math_o1's baseline_rl model,
# and all sql_* tasks reuse sql_conceptual's baseline_rl model.
if [[ "${TASK}" == "math" ]]; then
  BASE_MODEL_DATA_NAME="math_o1"
elif [[ "${TASK}" == "sql" ]]; then
  BASE_MODEL_DATA_NAME="sql_conceptual"
else
  BASE_MODEL_DATA_NAME="${DATA_NAME}"
fi

python -m pipeline train_rl \
  --task "${TASK}" \
  --method method_d \
  --run-id "${RUN_ID}" \
  --base-model "models/${BASE_MODEL_DATA_NAME}/baseline_rl/${RUN_ID}/model" \
  --data-name "${DATA_NAME}" \
  --n-samples 8 \
  --sft-stride 128 \
  --test-freq 25 \
  --keep-checkpoints \
  "${@:7}"
