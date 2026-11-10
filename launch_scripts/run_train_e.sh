#!/bin/bash
# Method E RL runner, initialized from the existing Method AC SFT model.
#
# Usage: bash run_train_e.sh <task> <model_short_name> <hf_base_model> <data_name>
# The third argument is retained for compatibility with run_train_ac.sh but unused.

set -euo pipefail

TASK="$1"
MODEL="$2"
DATA_NAME="$4"
RUN_ID="${MODEL}"

PROMPTS_DIR="data/${DATA_NAME}/problems_with_format"
NUM_HINTS="${NUM_HINTS:-5}"
for SPLIT in sft_train sft_val rl_train rl_val eval rl_gen_train rl_gen_val rl_ver_train rl_ver_val; do
  EXT="json"
  if [[ "${SPLIT}" == rl* ]]; then
    EXT="parquet"
  fi
  if [[ ! -f "${PROMPTS_DIR}/${SPLIT}__method_e.${EXT}" ]]; then
    echo "Missing Method E prompts for ${SPLIT}; creating all splits."
    # Match AC's maximum hint level; override with NUM_HINTS if needed.
    python -m pipeline create_prompts \
      --task "${TASK}" \
      --method method_e \
      --data-name "${DATA_NAME}" \
      --split all \
      --num-hints "${NUM_HINTS}"
    break
  fi
done

# Method E adds prefix-success prediction to the baseline RL objective:
# --success-loss-weight: multiplier on the auxiliary binary cross-entropy
#   loss. Total loss = RL loss + weight * success BCE; 0 disables its contribution.
# --success-interval: predict final rollout correctness after every L valid
#   generated response tokens (50 below); prompt and padding tokens do not count.
# --success-update-backbone: auxiliary BCE trains both the success head and
#   backbone. Use --no-success-update-backbone to train only the head with BCE;
#   the RL objective still updates the backbone in either mode.
python -m pipeline train_rl \
  --task "${TASK}" \
  --method method_e \
  --run-id "${RUN_ID}" \
  --sft-model "models/${DATA_NAME}/method_ac_sft/${RUN_ID}/model" \
  --data-name "${DATA_NAME}" \
  --success-loss-weight 0.1 \
  --success-interval 50 \
  --success-update-backbone \
  --overwrite \
  --override actor_rollout_ref.rollout.val_kwargs.do_sample=True \
             actor_rollout_ref.rollout.val_kwargs.temperature=1 \
             actor_rollout_ref.rollout.val_kwargs.n=1 \
             data.filter_overlong_prompts=True \
             data.filter_overlong_prompts_workers=8
