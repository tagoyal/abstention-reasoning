#!/usr/bin/env bash
set -euo pipefail

MODEL="qwen2.5-1.5b"
TASK="competition_math"

# Step 1: generate representative solver rollouts from method_ac. Already done
# once per model for the method_a verifier runs -- reuse those files instead of
# regenerating. Uncomment if they don't exist yet for this MODEL.
# python -m pipeline generate \
#   --task "${TASK}" \
#   --method method_ac \
#   --model rl \
#   --run-id "${MODEL}" \
#   --split sft_train \
#   --num-samples 10 \
#   --sample-strategy random \
#   --async \
#   --output "artifacts/${TASK}/sft_datasets/sft_train__method_ac__verification-10s__${MODEL}.internal.json"

# python -m pipeline generate \
#   --task "${TASK}" \
#   --method method_ac \
#   --model rl \
#   --run-id "${MODEL}" \
#   --split sft_val \
#   --num-samples 10 \
#   --sample-strategy random \
#   --async \
#   --output "artifacts/${TASK}/sft_datasets/sft_val__method_ac__verification-10s__${MODEL}.internal.json"

# Step 2: build method_c's verifier prompts from those solver rollouts.
# Ground truth is just whether the representative rollout was correct (1/0),
# Note that we generate using the sft dataset about, for both sft_train and sft_val. So, now sft_train is split into sft and rl datasets in 10/90 ratio. same for val. 
python -m pipeline create_verification_data \
  --task "${TASK}" \
  --method method_c \
  --generations "artifacts/${TASK}/sft_datasets/sft_train__method_ac__verification-10s__${MODEL}.internal.json" \
  --run-id "${MODEL}" \
  --split train

python -m pipeline create_verification_data \
  --task "${TASK}" \
  --method method_c \
  --generations "artifacts/${TASK}/sft_datasets/sft_val__method_ac__verification-10s__${MODEL}.internal.json" \
  --run-id "${MODEL}" \
  --split val

# Step 3: generate the verifier's own <think>/<answer> judgments on the SFT
# slice with Qwen3-14B; these become the SFT training targets. Sample 8 and
# keep the shortest correct judgment (falls back to shortest incorrect if
# none land on the correct label), rather than a single shot.
python -m pipeline generate \
  --task "${TASK}" \
  --method method_c \
  --model Qwen/Qwen3-14B \
  --run-id "${MODEL}" \
  --prompts "artifacts/${TASK}/problems_with_format/sft_train__method_c__${MODEL}.json" \
  --output "artifacts/${TASK}/sft_datasets/sft_train__method_c__${MODEL}.json" \
  --num-samples 8 \
  --sample-strategy random_correct \
  --async


# Step 4: SFT warm-start the verifier on those judgments.
python -m pipeline train_sft \
  --task "${TASK}" \
  --method method_c \
  --run-id "${MODEL}" \
  --base-model "artifacts/${TASK}/models/method_ac_models/${MODEL}/model" \
  --dataset "artifacts/${TASK}/sft_datasets/sft_train__method_c__${MODEL}.json" \
  --output "artifacts/${TASK}/models/method_c_predictors_sft/${MODEL}/model" \
  --completion-only-loss 

# Step 5: RL-train the verifier on the remaining RL slice, validating against
# the rl_val slice built in step 2.
python -m pipeline train_rl \
  --task "${TASK}" \
  --method method_c \
  --run-id "${MODEL}" \
  --sft-model "artifacts/${TASK}/models/method_c_predictors_sft/${MODEL}/model" \
  --train-prompts "artifacts/${TASK}/problems_with_format/rl_train__method_c__${MODEL}.parquet" \
  --val-prompts "artifacts/${TASK}/problems_with_format/rl_val__method_c__${MODEL}.parquet" \
  --overwrite

