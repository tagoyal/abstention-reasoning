#!/bin/bash
# Generic train_classifier runner for one (task, data_name) combo -- cycles
# through all run-ids (baseline_rl models), training a LoRA depth-accuracy
# classifier on top of each one. Output path is explicitly suffixed with
# "_b" (run_id itself is left as-is, since it's also used to resolve the
# default dataset/base-model paths) so this doesn't overwrite the earlier
# models/.../baseline_classifier/<run_id>/model runs.
#
# Usage: bash run_jobs_temp.sh <task> <data_name>

set -euo pipefail

TASK="$1"
DATA_NAME="$2"

RUN_IDS=(qwen2.5-1.5b qwen2.5-3b qwen3-4b qwen3-8b)

for RUN_ID in "${RUN_IDS[@]}"; do
  python -m pipeline train_classifier \
    --task "${TASK}" --method baseline --run-id "${RUN_ID}" \
    --base-model "models/${DATA_NAME}/baseline_rl/${RUN_ID}/model" \
    --data-name "${DATA_NAME}" \
    --output "models/${DATA_NAME}/baseline_classifier/${RUN_ID}_b/model" \
    --use-lora \
    --balance-train \
    --epochs 3 \
    --max-length 2048 \
    --depth-eval-max-new-tokens 1
done
