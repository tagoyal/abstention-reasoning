#!/bin/bash
# SFT + RL for method_b_composed.
# Usage: bash launch_scripts/run_train_composer_b.sh <task> <model_short_name> <hf_base_model> <data_name> [success_update_backbone] [compose_k] [alpha] [base] [sft_data_desc]
# Arguments 5 and 9 are unused; this runner uses unsuffixed SFT datasets.
# Set SFT_MODEL=<path> to skip SFT and start RL from an existing checkpoint.

set -euo pipefail

TASK="$1"
MODEL="$2"
HF_MODEL="$3"
DATA_NAME="$4"
COMPOSE_K="${6:-${COMPOSE_K:-2}}"
ALPHA="${7:-0.5}"
BASE="${8:-0.1}"
METHOD="method_b_composed"
RUN_ID="${MODEL}-k${COMPOSE_K}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$((2048 * COMPOSE_K))}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-$((MAX_RESPONSE_LENGTH + 2048))}"
COMPOSE_AGGREGATE="${COMPOSE_AGGREGATE:-min}"

STRIP_THINK=""
if [[ "${MODEL,,}" == *qwen3* ]]; then
  STRIP_THINK="--strip-think-tokens"
fi

if [[ -z "${SFT_MODEL:-}" ]]; then
  SFT_MODEL="models/${DATA_NAME}/${METHOD}_sft/${RUN_ID}/model"
  SFT_STEM="data/${DATA_NAME}/sft_datasets"
  python -m pipeline train_sft \
    --task "${TASK}" \
    --method "${METHOD}" \
    --compose-k "${COMPOSE_K}" \
    --run-id "${RUN_ID}" \
    --base-model "${HF_MODEL}" \
    --data-name "${DATA_NAME}" \
    --max-length "${SFT_MAX_LENGTH}" \
    --dataset "${SFT_STEM}/sft_train__${METHOD}__k${COMPOSE_K}.json" \
    --eval-dataset "${SFT_STEM}/sft_val__${METHOD}__k${COMPOSE_K}.json" \
    ${STRIP_THINK}
fi

PROMPTS_DIR="data/${DATA_NAME}/problems_with_format"
if [ ! -f "${PROMPTS_DIR}/rl_train__${METHOD}__k${COMPOSE_K}.parquet" ] || \
   [ ! -f "${PROMPTS_DIR}/rl_val__${METHOD}__k${COMPOSE_K}.parquet" ]; then
  python -m pipeline create_prompts --task "${TASK}" --method "${METHOD}" \
    --compose-k "${COMPOSE_K}" --split all --data-name "${DATA_NAME}"
fi

python -m pipeline train_rl \
  --task "${TASK}" \
  --method "${METHOD}" \
  --compose-k "${COMPOSE_K}" \
  --run-id "${RUN_ID}_quad-a${ALPHA}_b${BASE}" \
  --sft-model "${SFT_MODEL}" \
  --data-name "${DATA_NAME}" \
  --max-response-length "${MAX_RESPONSE_LENGTH}" \
  --overwrite \
  --reward-kwargs compose_aggregate="${COMPOSE_AGGREGATE}" \
                 hint_penalty=0.1 hint_penalty_shape=quadratic \
                 hint_penalty_alpha="${ALPHA}" hint_penalty_base="${BASE}" \
  --override actor_rollout_ref.rollout.val_kwargs.do_sample=True \
             actor_rollout_ref.rollout.val_kwargs.temperature=1 \
             actor_rollout_ref.rollout.val_kwargs.n=1
