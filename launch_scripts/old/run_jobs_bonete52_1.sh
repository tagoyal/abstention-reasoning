#python -m pipeline generate --task countdown --method method_ac --model Qwen/Qwen3-14B --split sft_train --num-samples 8 --sample-strategy random_correct --async


for TASK in countdown math; do
  for MODEL in qwen2.5-3b qwen3-4b-base; do

  # python -m pipeline generate \
  # --task "${TASK}" \
  # --method baseline \
  # --model rl \
  # --run-id "${MODEL}" \
  # --split eval \
  # --num-samples 10 \
  # --sample-strategy random \
  # --async \
  # --output "artifacts/${TASK}/sft_datasets_verification/eval__baseline__verification-10s__${MODEL}.internal.json"
  
  python -m pipeline create_verification_data \
  --task "${TASK}" \
  --method method_a \
  --generations "artifacts/${TASK}/sft_datasets_verification/eval__baseline__verification-10s__${MODEL}.internal.json" \
  --output "artifacts/${TASK}/sft_datasets_verification/eval__baseline__qh_predictions__${MODEL}.json"

  done
done