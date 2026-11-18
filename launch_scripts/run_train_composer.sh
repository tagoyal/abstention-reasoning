#!/bin/bash
# train_sft + train_rl runner for the composed baseline (K problems per datapoint),
# for one (task, model) combo. Same layout as run_train.sh.
#
# Usage: bash run_train_composer.sh <task> <model_short_name> <hf_base_model> <data_name> [compose_k]
#   e.g. bash run_train_composer.sh countdown qwen2.5-1.5b Qwen/Qwen2.5-1.5B countdown 2
#
# Needs data/{data_name}/sft_datasets/sft_{train,val}__baseline_composed__k{K}.json
# (run_sft_data.sh) and the rl_{train,val} composed parquets (create_prompts).

set -euo pipefail

TASK="$1"
MODEL="$2"
HF_MODEL="$3"
DATA_NAME="$4"
COMPOSE_K="${5:-${COMPOSE_K:-2}}"
METHOD="${METHOD:-baseline_composed}"
# Model dirs are keyed by method + run id, not K, so K goes in the run id.
RUN_ID="${MODEL}-k${COMPOSE_K}"
# Baseline uses 2048 response tokens; one composed response answers all K subparts.
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$((2048 * COMPOSE_K))}"
# SFT sequences hold the prompt (K problems) plus the response.
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-$((MAX_RESPONSE_LENGTH + 2048))}"
# "min": full reward only if every subpart is correct; "mean": partial credit per subpart.
COMPOSE_AGGREGATE="${COMPOSE_AGGREGATE:-min}"

# See run_train.sh: Qwen3 needs its <think> special tokens stripped.
STRIP_THINK=""
if [[ "${MODEL,,}" == *qwen3* ]]; then
  STRIP_THINK="--strip-think-tokens"
fi

python -m pipeline train_sft \
  --task "${TASK}" \
  --method "${METHOD}" \
  --compose-k "${COMPOSE_K}" \
  --run-id "${RUN_ID}" \
  --base-model "${HF_MODEL}" \
  --data-name "${DATA_NAME}" \
  --max-length "${SFT_MAX_LENGTH}" \
  ${STRIP_THINK}

python -m pipeline train_rl \
  --task "${TASK}" \
  --method "${METHOD}" \
  --compose-k "${COMPOSE_K}" \
  --run-id "${RUN_ID}" \
  --sft-model "models/${DATA_NAME}/${METHOD}_sft/${RUN_ID}/model" \
  --data-name "${DATA_NAME}" \
  --max-response-length "${MAX_RESPONSE_LENGTH}" \
  --overwrite \
  --reward-kwargs compose_aggregate="${COMPOSE_AGGREGATE}" \
  --override actor_rollout_ref.rollout.val_kwargs.do_sample=True \
             actor_rollout_ref.rollout.val_kwargs.temperature=1 \
             actor_rollout_ref.rollout.val_kwargs.n=1
