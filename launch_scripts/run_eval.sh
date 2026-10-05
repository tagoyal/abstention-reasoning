#!/bin/bash
# Generic evaluation runner for one (task, model) combo, evaluating the
# baseline method's RL model.
#
# Usage: bash run_eval.sh <task> <model_short_name> <data_name>
#   e.g. bash run_eval.sh countdown qwen2.5-1.5b countdown

set -euo pipefail

TASK="$1"
MODEL="$2"
DATA_NAME="$3"
RUN_ID="${MODEL}"

python -m pipeline evaluate \
  --task "${TASK}" \
  --method baseline \
  --run-id "${RUN_ID}" \
  --data-name "${DATA_NAME}" \
  --model rl \
  --async \
  --num-samples 10
