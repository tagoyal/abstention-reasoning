#!/bin/bash
# SFT + RL for method_ac_composed, initialized from the latest dated baseline_rl run.
# Usage: bash launch_scripts/run_train_composer_ac.sh <task> <model_short_name> <hf_base_model> <data_name> [success_update_backbone] [compose_k] [alpha] [base] [sft_data_desc]
# Arguments 3, 5, 7, 8 and 9 are unused; SFT uses the stitched datasets.
# Set SFT_MODEL=<path> to skip SFT and start RL from an existing checkpoint.

set -euo pipefail

TASK="$1"
MODEL="$2"
DATA_NAME="$4"
COMPOSE_K="${6:-${COMPOSE_K:-2}}"
METHOD="method_ac_composed"
RUN_ID="${MODEL}-k${COMPOSE_K}-stitched"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$((2048 * COMPOSE_K))}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-$((MAX_RESPONSE_LENGTH + 2048))}"
COMPOSE_AGGREGATE="${COMPOSE_AGGREGATE:-min}"
RL_RUN_ID="${RUN_ID}"
if [[ "${COMPOSE_AGGREGATE}" == "mean" ]]; then
  RL_RUN_ID="${RUN_ID}-mean"
fi
case "${TASK}" in
  math|countdown) ;;
  *) echo "method_ac_composed supports only math and countdown." >&2; exit 1 ;;
esac

STRIP_THINK=()
if [[ "${MODEL,,}" == *qwen3* ]]; then
  STRIP_THINK=(--strip-think-tokens)
fi

if [[ -z "${SFT_MODEL:-}" ]]; then
  # Dated run names use MM_DD-baseline-<model>; glob order follows the date.
  export LC_ALL=C
  shopt -s nullglob
  BASELINE_MODELS=(models/"${DATA_NAME}"/baseline_rl/[0-1][0-9]_[0-3][0-9]-baseline-"${MODEL}"/model)
  shopt -u nullglob
  if [[ ${#BASELINE_MODELS[@]} -eq 0 ]]; then
    echo "No dated baseline_rl checkpoint found for ${MODEL} in models/${DATA_NAME}/baseline_rl." >&2
    exit 1
  fi
  BASELINE_MODEL="${BASELINE_MODELS[${#BASELINE_MODELS[@]}-1]}"
  if [[ ! -f "${BASELINE_MODEL}/config.json" ]]; then
    echo "Latest baseline RL checkpoint not found: ${BASELINE_MODEL}" >&2
    exit 1
  fi
  echo "Initializing SFT from ${BASELINE_MODEL}"
  SFT_MODEL="models/${DATA_NAME}/${METHOD}_sft/${RUN_ID}/model"
  SFT_STEM="data/${DATA_NAME}/sft_datasets"
  python -m pipeline train_sft \
    --task "${TASK}" --method "${METHOD}" --compose-k "${COMPOSE_K}" \
    --run-id "${RUN_ID}" --base-model "${BASELINE_MODEL}" \
    --data-name "${DATA_NAME}" --max-length "${SFT_MAX_LENGTH}" \
    --dataset "${SFT_STEM}/sft_train__${METHOD}__k${COMPOSE_K}__stitched.json" \
    --eval-dataset "${SFT_STEM}/sft_val__${METHOD}__k${COMPOSE_K}__stitched.json" \
    "${STRIP_THINK[@]}"
fi

PROMPTS_DIR="data/${DATA_NAME}/problems_with_format"
for SPLIT in rl_train rl_val; do
  if [[ ! -f "${PROMPTS_DIR}/${SPLIT}__${METHOD}__k${COMPOSE_K}.parquet" ]] &&
     ! compgen -G "${PROMPTS_DIR}/${SPLIT}__${METHOD}__k${COMPOSE_K}.shard*.parquet" > /dev/null; then
    python -m pipeline create_prompts --task "${TASK}" --method "${METHOD}" \
      --compose-k "${COMPOSE_K}" --split "${SPLIT}" --data-name "${DATA_NAME}"
  fi
done

python -m pipeline train_rl \
  --task "${TASK}" --method "${METHOD}" --compose-k "${COMPOSE_K}" \
  --run-id "${RL_RUN_ID}" --sft-model "${SFT_MODEL}" \
  --data-name "${DATA_NAME}" --max-response-length "${MAX_RESPONSE_LENGTH}" \
  --overwrite --reward-kwargs compose_aggregate="${COMPOSE_AGGREGATE}" \
  --override actor_rollout_ref.rollout.val_kwargs.do_sample=True \
             actor_rollout_ref.rollout.val_kwargs.temperature=1 \
             actor_rollout_ref.rollout.val_kwargs.n=1
