#!/bin/bash
# Generic train_sft + train_rl runner for one (task, model) combo.
# Reads from the v2 (weighted hint-level) dataset/prompts, and writes to a
# "_v2" run-id so outputs don't collide with runs already produced from the
# original (non-v2) data.
#
# Usage: bash run_train.sh <task> <model>
#   e.g. bash run_train.sh countdown qwen2.5-1.5b

set -euo pipefail

TASK="$1"
MODEL="$2"
RUN_ID="${MODEL}_v3"

python -m pipeline train_sft \
  --task "${TASK}" \
  --method method_ac \
  --run-id "${RUN_ID}" \
  --base-model "artifacts/${TASK}/models/baseline_models/${MODEL}/model" \
  --dataset "artifacts/${TASK}/sft_datasets/sft_train__method_ac_v3.json"

python -m pipeline train_rl \
  --task "${TASK}" \
  --method method_ac \
  --run-id "${RUN_ID}" \
  --sft-model "artifacts/${TASK}/models/method_ac_sft/${RUN_ID}/model" \
  --train-prompts "artifacts/${TASK}/problems_with_format_v3/rl_train__method_ac.parquet" \
  --val-prompts "artifacts/${TASK}/problems_with_format_v3/rl_val__method_ac.parquet" \
  --overwrite
