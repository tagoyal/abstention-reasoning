

set -euo pipefail

TASK="$1"
MODEL="$2"
RUN_ID="${MODEL}_v3"

python -m pipeline create_verification_data \
  --task "${TASK}" \
  --method method_a \
  --generations "artifacts/${TASK}/sft_datasets/sft_train__method_ac__verification-10s__${RUN_ID}.internal.json" \
  --output "artifacts/${TASK}/sft_datasets/sft_train__method_a__qh_predictions__${RUN_ID}.json"

python -m pipeline create_verification_data \
  --task "${TASK}" \
  --method method_a \
  --generations "artifacts/${TASK}/sft_datasets/sft_val__method_ac__verification-10s__${RUN_ID}.internal.json" \
  --output "artifacts/${TASK}/sft_datasets/sft_val__method_a__qh_predictions__${RUN_ID}.json"


python -m pipeline train_sft \
  --task "${TASK}" \
  --method method_a \
  --run-id "${RUN_ID}" \
  --base-model "artifacts/${TASK}/models/method_ac_models/${RUN_ID}/model" \
  --dataset "artifacts/${TASK}/sft_datasets/sft_train__method_a__qh_predictions__${RUN_ID}.json" \
  --output "artifacts/${TASK}/models/method_a_predictors/${RUN_ID}/model" \
  --completion-only-loss
