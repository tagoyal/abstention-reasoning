#!/bin/bash
# train_sft + train_rl runner for the composed baseline (K problems per datapoint),
# for one (task, model) combo. Same layout as run_train.sh.
#
# Usage: bash run_train_composer.sh <task> <model_short_name> <hf_base_model> <data_name> [success_update_backbone] [compose_k] [alpha] [base] [sft_data_desc]
#   e.g. bash run_train_composer.sh countdown qwen2.5-1.5b Qwen/Qwen2.5-1.5B countdown true 2
# Arguments 5, 7 and 8 are unused.
#
# Needs data/{data_name}/sft_datasets/sft_{train,val}__baseline_composed__k{K}.json
# (run_sft_data.sh) and the rl_{train,val} composed parquets (create_prompts).
#
# SFT_DATA_DESC=stitched trains on {split}__{method}__k{K}__stitched.json (MODE=composed_stitched
# run_sft_data.sh) instead; the desc is appended to the run id so models don't collide.
#
# Other composed methods: set METHOD (e.g. METHOD=method_b_composed, whose SFT
# data comes from MODE=composed_hints run_sft_data.sh). Set SFT_MODEL=<path> to
# skip train_sft and start RL from an existing checkpoint:
#   METHOD=method_b_composed SFT_MODEL=models/math_o3/method_b_composed_sft/qwen2.5-3b-k2/model \
#     bash run_train_composer.sh math qwen2.5-3b Qwen/Qwen2.5-3B math_o3 true 2

set -euo pipefail

TASK="$1"
MODEL="$2"
HF_MODEL="$3"
DATA_NAME="$4"
COMPOSE_K="${6:-${COMPOSE_K:-2}}"
METHOD="${METHOD:-baseline_composed}"
SFT_DATA_DESC="${9-${SFT_DATA_DESC-stitched}}"
# Model dirs are keyed by method + run id, not K, so K goes in the run id.
RUN_ID="${MODEL}-k${COMPOSE_K}${SFT_DATA_DESC:+-${SFT_DATA_DESC}}"
SFT_DATA_ARGS=()
if [[ -n "${SFT_DATA_DESC:-}" ]]; then
  SFT_STEM="data/${DATA_NAME}/sft_datasets"
  SFT_DATA_ARGS=(--dataset "${SFT_STEM}/sft_train__${METHOD}__k${COMPOSE_K}__${SFT_DATA_DESC}.json"
                 --eval-dataset "${SFT_STEM}/sft_val__${METHOD}__k${COMPOSE_K}__${SFT_DATA_DESC}.json")
fi
# Baseline uses 2048 response tokens; one composed response answers all K subparts.
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$((2048 * COMPOSE_K))}"
# SFT sequences hold the prompt (K problems) plus the response.
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-$((MAX_RESPONSE_LENGTH + 2048))}"
# "min": full reward only if every subpart is correct; "mean": partial credit per subpart.
COMPOSE_AGGREGATE="${COMPOSE_AGGREGATE:-mean}"
RL_RUN_ID="${RUN_ID}"
if [[ "${COMPOSE_AGGREGATE}" == "mean" ]]; then
  RL_RUN_ID="${RUN_ID}-mean"
fi

# See run_train.sh: Qwen3 needs its <think> special tokens stripped.
STRIP_THINK=""
if [[ "${MODEL,,}" == *qwen3* ]]; then
  STRIP_THINK="--strip-think-tokens"
fi

if [[ -z "${SFT_MODEL:-}" ]]; then
  SFT_MODEL="models/${DATA_NAME}/${METHOD}_sft/${RUN_ID}/model"
  python -m pipeline train_sft \
    --task "${TASK}" \
    --method "${METHOD}" \
    --compose-k "${COMPOSE_K}" \
    --run-id "${RUN_ID}" \
    --base-model "${HF_MODEL}" \
    --data-name "${DATA_NAME}" \
    --max-length "${SFT_MAX_LENGTH}" \
    "${SFT_DATA_ARGS[@]}" \
    ${STRIP_THINK}
fi

# RL prompts for this method/K, built if missing (partitions are untouched).
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
  --run-id "${RL_RUN_ID}" \
  --sft-model "${SFT_MODEL}" \
  --data-name "${DATA_NAME}" \
  --max-response-length "${MAX_RESPONSE_LENGTH}" \
  --overwrite \
  --reward-kwargs compose_aggregate="${COMPOSE_AGGREGATE}" \
  --override actor_rollout_ref.rollout.val_kwargs.do_sample=True \
             actor_rollout_ref.rollout.val_kwargs.temperature=1 \
             actor_rollout_ref.rollout.val_kwargs.n=1
