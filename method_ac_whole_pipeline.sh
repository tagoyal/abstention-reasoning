# for TASK in countdown competition_math; do
#   python -m pipeline create_prompts --task "$TASK" --method method_ac --split all --num-hints 5 \
#     --output "artifacts/${TASK}/problems_with_format_v2"
# done


for TASK in countdown competition_math; do

  python -m pipeline generate --task "${TASK}" --method method_ac --model Qwen/Qwen3-14B \
  --split sft_train --num-samples 8 --sample-strategy random_correct --async \
  --prompts artifacts/${TASK}/problems_with_format_v2/sft_train__method_ac.json \
  --output artifacts/${TASK}/sft_datasets/sft_train__method_ac_v2.json

done


