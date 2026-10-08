#!/bin/bash
# Generic evaluation runner for one (task, model) combo, evaluating the
# baseline method's RL model.
#
# Usage: bash run_eval.sh <task> <model_short_name> <data_name> [gpus]
#   e.g. bash run_eval.sh countdown qwen2.5-1.5b countdown 2

set -euo pipefail

TASK="$1"
MODEL="$2"
DATA_NAME="$3"
GPUS="${4:-1}"
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

