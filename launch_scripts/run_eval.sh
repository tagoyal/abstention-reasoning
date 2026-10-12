#!/bin/bash
# Generic evaluation runner for one (task, model) combo, evaluating the
# baseline method's RL model.
#
# Usage: bash run_eval.sh <task> <model_short_name> <data_name> [gpus] [alpha] [base]
#   e.g. bash run_eval.sh sql qwen2.5-1.5b sql_partial_sols 1 0.5 0.05
# alpha/base must match the hint_penalty_alpha/hint_penalty_base used in
# run_train_b.sh; they select the RL run id <model>_quad-a<alpha>_b<base>.

set -euo pipefail

TASK="$1"
MODEL="$2"
DATA_NAME="$3"
GPUS="${4:-1}"
ALPHA="${5:-0.5}"
BASE="${6:-0.1}"
RUN_ID="${MODEL}"

python -m pipeline evaluate \
  --task "${TASK}" \
  --method method_ac \
  --run-id "${RUN_ID}" \
  --data-name "${DATA_NAME}" \
  --model rl \
  --data-parallel-size "${GPUS}" \
  --async \
  --num-samples 10

# for SPLIT in rl_ver_train rl_ver_val; do
#   python -m pipeline generate_tree \
#     --task "${TASK}" \
#     --method baseline \
#     --run-id "${RUN_ID}" \
#     --data-name "${DATA_NAME}" \
#     --model rl \
#     --split "${SPLIT}" \
#     --num-midpoints 2 \
#     --num-samples 10 \
#     --data-parallel-size "${GPUS}" \
#     --async
# done

