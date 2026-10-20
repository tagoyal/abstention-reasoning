#!/bin/bash
# Generic train_sft + train_rl runner for one (task, model) combo, using the
# baseline method and the math_o1 data/models namespace.
#
# Usage: bash run_train_b.sh <task> <model_short_name> <hf_base_model> <data_name> [alpha] [base]
#   e.g. bash run_train_b.sh math qwen2.5-1.5b Qwen/Qwen2.5-1.5B sql_conceptual 0.5 0.1
# alpha (default 0.5) is hint_penalty_alpha for the quadratic hint penalty and
# base (default 0.1) is hint_penalty_base, a fixed cost once any hint is used.
# Both are embedded in the RL run id as _quad-a<alpha>_b<base>.

set -euo pipefail

TASK="$1"
MODEL="$2"
HF_MODEL="$3"
DATA_NAME="$4"
ALPHA="${5:-0.5}"
BASE="${6:-0.1}"
RUN_ID="${MODEL}"

# Qwen3 ships <think>/</think> as atomic special tokens (151667/151668) that a
# base model has no pretrained prior for, so it rarely emits </think> and RL
# burns its reward budget recovering format instead of learning to reason.
# Stripping the added-token entries makes Qwen3 tokenize the tags as ordinary
# text (as Qwen2.5 already does); the stripped tokenizer is saved with the SFT
# model and inherited by the downstream RL run. Qwen2.5 needs no such flag.
STRIP_THINK=""
if [[ "${MODEL,,}" == *qwen3* ]]; then
  STRIP_THINK="--strip-think-tokens"
fi

# The baseline_rl model used to seed train_sft lives under a fixed
# data/model namespace per task family, regardless of which DATA_NAME
# variant is being trained on: math_o2 reuses math_o1's baseline_rl model,
# and all sql_* tasks reuse sql_conceptual's baseline_rl model.
if [[ "${TASK}" == "math" ]]; then
  BASE_MODEL_DATA_NAME="math_o1"
elif [[ "${TASK}" == "sql" ]]; then
  BASE_MODEL_DATA_NAME="sql_conceptual"
else
  BASE_MODEL_DATA_NAME="${DATA_NAME}"
fi

# run_sft_data.sh writes sft_{train,val}__method_b__<SUFFIX>.json; the default
# lookup expects no suffix, so point train_sft at the suffixed files.
SUFFIX="${SUFFIX:-hintmix}"
DS_DIR="data/${DATA_NAME}/sft_datasets"
TRAIN_SUFFIX="${SUFFIX}"
if [[ "${DATA_NAME}" == "sql_2" ]]; then
  TRAIN_SUFFIX="${SUFFIX}_upsampled"
fi

python -m pipeline train_sft \
  --task "${TASK}" \
  --method method_b \
  --run-id "${RUN_ID}" \
  --base-model models/${BASE_MODEL_DATA_NAME}/baseline_rl/${RUN_ID}/model \
  --data-name "${DATA_NAME}" \
  --dataset "${DS_DIR}/sft_train__method_b__${TRAIN_SUFFIX}.json" \
  --eval-dataset "${DS_DIR}/sft_val__method_b__${SUFFIX}.json" \
  ${STRIP_THINK}

python -m pipeline train_rl \
  --task "${TASK}" \
  --method method_b \
  --run-id "${RUN_ID}"_quad-a"${ALPHA}"_b"${BASE}" \
  --sft-model "models/${DATA_NAME}/method_b_sft/${RUN_ID}/model" \
  --data-name "${DATA_NAME}" \
  --overwrite \
  --save-freq 50 \
  --test-freq 25 \
  --keep-checkpoints \
  --reward-kwargs hint_penalty=0.1 hint_penalty_shape=quadratic hint_penalty_alpha=${ALPHA} hint_penalty_base=${BASE} \
  --override trainer.max_actor_ckpt_to_keep=null \
  --override algorithm.norm_adv_by_std_in_grpo=False \
  --override actor_rollout_ref.rollout.val_kwargs.do_sample=True \
             actor_rollout_ref.rollout.val_kwargs.temperature=1 \
             actor_rollout_ref.rollout.val_kwargs.n=1
             # add actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean above to change loss aggregation