#!/bin/bash
# Generic train_sft + train_rl runner for one (task, model) combo, using the
# baseline method and the math_o1 data/models namespace.
#
# Usage: bash run_train.sh <task> <model_short_name> <hf_base_model> <data_name>
#   e.g. bash run_train.sh math qwen2.5-1.5b Qwen/Qwen2.5-1.5B sql_conceptual

set -euo pipefail

TASK="$1"
MODEL="$2"
HF_MODEL="$3"
DATA_NAME="$4"
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

python -m pipeline train_sft \
  --task "${TASK}" \
  --method method_ac \
  --run-id "${RUN_ID}" \
  --base-model models/${DATA_NAME}/baseline_rl/${RUN_ID}/model \
  --data-name "${DATA_NAME}" \
  ${STRIP_THINK}

python -m pipeline train_rl \
  --task "${TASK}" \
  --method method_ac \
  --run-id "${RUN_ID}" \
  --sft-model "models/${DATA_NAME}/method_ac_sft/${RUN_ID}/model" \
  --data-name "${DATA_NAME}" \
  --overwrite  \
  --override actor_rollout_ref.rollout.val_kwargs.do_sample=True \
             actor_rollout_ref.rollout.val_kwargs.temperature=1 \
             actor_rollout_ref.rollout.val_kwargs.n=1
             # add actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean above to change loss aggregation


