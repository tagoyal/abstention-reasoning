#!/bin/bash
# Usage: bash launch_scripts/run_sft_data.sh <task> [gen_model]
# Launched per task by launch_sft_data_jobs.sh.
set -euo pipefail

TASK="$1"
GEN_MODEL="${2:-Qwen/Qwen3-14B}"
# Exact share of examples per hint count. Each problem is capped at its count.
HINT_MIX="${HINT_MIX:-0:0.4,1:0.2,2:0.15,3:0.1,4:0.08,5:0.07}"
# Suffix for the output files: <split>__method_b__<SUFFIX>.json. Train on them with --dataset.
SUFFIX="${SUFFIX:-hintmix}"

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



for SPLIT in sft_train sft_val; do
  # generate resumes from existing outputs, so remove them to regenerate from scratch
  rm -f "data/${DATA_NAME}/.scratch/${SPLIT}__baseline__probe.json" \
        "data/${DATA_NAME}/sft_datasets/${SPLIT}__method_b__${SUFFIX}.json"

  python -m pipeline generate --task "${TASK}" --method baseline --async \
    --model "${GEN_MODEL}" --split "${SPLIT}" --no-hints --num-samples 8 \
    --data-name "${DATA_NAME}" \
    --output data/${DATA_NAME}/.scratch/${SPLIT}__baseline__probe.json

  # Schedule hints from that probe -- harder problems get more -- and oversample
  # until the worst hint bucket is half correct. The profile is derived in memory; only the
  # dataset is written.
  python -m pipeline generate --task "${TASK}" --method method_b --async \
      --model "${GEN_MODEL}" --split "${SPLIT}" \
      --hint-schedule data/${DATA_NAME}/.scratch/${SPLIT}__baseline__probe.json \
      --hint-mix "${HINT_MIX}" --target-correct-rate 0.5 \
      --output "data/${DATA_NAME}/sft_datasets/${SPLIT}__method_b__${SUFFIX}.json" \
      --data-name "${DATA_NAME}"
done

# python -m pipeline generate --task "${TASK}" --data-name "${DATA_NAME}" --method method_ac --model "${GEN_MODEL}" \
# --split sft_train --num-samples 8 --sample-strategy random_correct --async 

# python -m pipeline generate --task "${TASK}" --data-name "${DATA_NAME}" --method method_ac --model "${GEN_MODEL}" \
# --split sft_val --num-samples 8 --sample-strategy random_correct --async 