#!/bin/bash
# Usage: bash launch_scripts/run_sft_data.sh <task> [gen_model]
# Launched per task by launch_sft_data_jobs.sh.
set -euo pipefail

TASK="$1"
GEN_MODEL="${2:-Qwen/Qwen3-14B}"

DATA_NAME="$TASK"
if [ "$TASK" = "sql" ]; then
  DATA_NAME="sql_partial_sols"
fi
if [ "$TASK" = "math" ]; then
  DATA_NAME="math_o2"
fi

# python -m pipeline generate --task sql --method baseline --async \
#     --model Qwen/Qwen3-32B --split sft_train \
#     --data-name sql_partial_sql --num-samples 10 \
#   --sample-strategy random 

# python -m pipeline generate --task sql --method baseline --async \
#     --model Qwen/Qwen3-32B --split sft_val \
#     --data-name sql_partial_sql --num-samples 10 \
#   --sample-strategy random

# python -m pipeline generate --task "${TASK}" --data-name "${DATA_NAME}" --method method_ac --model "${GEN_MODEL}" \
# --split sft_train --num-samples 8 --sample-strategy random_correct --async 

# python -m pipeline generate --task "${TASK}" --data-name "${DATA_NAME}" --method method_ac --model "${GEN_MODEL}" \
# --split sft_val --num-samples 8 --sample-strategy random_correct --async 

for SPLIT in sft_train sft_val; do
  python -m pipeline generate --task "${TASK}" --method baseline --async \
    --model "${GEN_MODEL}" --split "${SPLIT}" --no-hints --num-samples 8 \
    --data-name "${DATA_NAME}" \
    --output data/${DATA_NAME}/.scratch/${SPLIT}__baseline__probe.json

  # Schedule hints from that probe -- harder problems get more -- and oversample
  # until half the set is correct. The profile is derived in memory; only the
  # dataset is written.
  python -m pipeline generate --task "${TASK}" --method method_b --async \
      --model "${GEN_MODEL}" --split "${SPLIT}" \
      --hint-schedule data/${DATA_NAME}/.scratch/${SPLIT}__baseline__probe.json \
      --hint-target-fraction 0.5 --max-hints 5 --target-correct-rate 0.5 \
      --data-name "${DATA_NAME}"
done
