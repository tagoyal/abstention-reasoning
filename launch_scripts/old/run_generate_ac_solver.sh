

set -euo pipefail

TASK="$1"
MODEL="$2"
RUN_ID="${MODEL}_v3"

python -m pipeline generate \
  --task "${TASK}" \
  --method method_ac \
  --model rl \
  --run-id "${RUN_ID}" \
  --split sft_train \
  --num-samples 10 \
  --sample-strategy random \
  --async \
  --output "artifacts/${TASK}/sft_datasets/sft_train__method_ac__verification-10s__${RUN_ID}.internal.json"

python -m pipeline generate \
  --task "${TASK}" \
  --method method_ac \
  --model rl \
  --run-id "${RUN_ID}" \
  --split sft_val \
  --num-samples 10 \
  --sample-strategy random \
  --async \
  --output "artifacts/${TASK}/sft_datasets/sft_val__method_ac__verification-10s__${RUN_ID}.internal.json"