

set -euo pipefail

TASK="$1"
MODEL="$2"
RUN_ID="${MODEL}_v3"

python -m pipeline create_verification_data \
  --task "${TASK}" \
  --method method_c \
  --generations "artifacts/${TASK}/sft_datasets/sft_train__method_ac__verification-10s__${RUN_ID}.internal.json" \
  --run-id "${RUN_ID}" \
  --split train

python -m pipeline create_verification_data \
  --task "${TASK}" \
  --method method_c \
  --generations "artifacts/${TASK}/sft_datasets/sft_val__method_ac__verification-10s__${RUN_ID}.internal.json" \
  --run-id "${RUN_ID}" \
  --split val

# Step 3: generate the verifier's own <think>/<answer> judgments on the SFT
# slice with Qwen3-14B; these become the SFT training targets. Sample 8 and
# keep the shortest correct judgment (falls back to shortest incorrect if
# none land on the correct label), rather than a single shot.
python -m pipeline generate \
  --task "${TASK}" \
  --method method_c \
  --model Qwen/Qwen3-14B \
  --run-id "${RUN_ID}" \
  --prompts "artifacts/${TASK}/problems_with_format/sft_train__method_c__${RUN_ID}.json" \
  --output "artifacts/${TASK}/sft_datasets/sft_train__method_c__${RUN_ID}.json" \
  --num-samples 8 \
  --sample-strategy random_correct \
  --async


# Step 4: SFT warm-start the verifier on those judgments.
python -m pipeline train_sft \
  --task "${TASK}" \
  --method method_c \
  --run-id "${RUN_ID}" \
  --base-model "artifacts/${TASK}/models/method_ac_models/${RUN_ID}/model" \
  --dataset "artifacts/${TASK}/sft_datasets/sft_train__method_c__${RUN_ID}.json" \
  --output "artifacts/${TASK}/models/method_c_predictors_sft/${RUN_ID}/model" \
  --completion-only-loss 

# Step 5: RL-train the verifier on the remaining RL slice, validating against
# the rl_val slice built in step 2.
python -m pipeline train_rl \
  --task "${TASK}" \
  --method method_c \
  --run-id "${RUN_ID}" \
  --sft-model "artifacts/${TASK}/models/method_c_predictors_sft/${RUN_ID}/model" \
  --train-prompts "artifacts/${TASK}/problems_with_format/rl_train__method_c__${RUN_ID}.parquet" \
  --val-prompts "artifacts/${TASK}/problems_with_format/rl_val__method_c__${RUN_ID}.parquet" \
  --overwrite