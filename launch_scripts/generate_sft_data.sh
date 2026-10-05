# python -m pipeline generate --task sql --method baseline --async \
#     --model Qwen/Qwen3-32B --split sft_train \
#     --data-name sql_partial_sql --num-samples 10 \
#   --sample-strategy random 

# python -m pipeline generate --task sql --method baseline --async \
#     --model Qwen/Qwen3-32B --split sft_val \
#     --data-name sql_partial_sql --num-samples 10 \
#   --sample-strategy random

for TASK in math; do
  DATA_NAME="$TASK"
  if [ "$TASK" = "sql" ]; then
    DATA_NAME="sql_conceptual"
  fi
  if [ "$TASK" = "math" ]; then
    DATA_NAME="math_o1"
  fi
  GEN_MODEL="Qwen/Qwen3-14B"
  if [ "$TASK" = "sql" ]; then
    GEN_MODEL="Qwen/Qwen3-32B"
  fi
  python -m pipeline generate --task "${TASK}" --data-name "${DATA_NAME}" --method method_ac --model "${GEN_MODEL}" \
  --split sft_train --num-samples 8 --sample-strategy random_correct --async 

  python -m pipeline generate --task "${TASK}" --data-name "${DATA_NAME}" --method method_ac --model "${GEN_MODEL}" \
  --split sft_val --num-samples 8 --sample-strategy random_correct --async 
done