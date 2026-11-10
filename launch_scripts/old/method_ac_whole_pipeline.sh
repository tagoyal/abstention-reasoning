#EVRYTHING MUST BE DONE PARALLELY, THIS IS JUST INFO

#CREATE THE FORMATS
for DATA_NAME in countdown math_o2 sql_partial_sols; do
  TASK="sql"
  if [ "$DATA_NAME" = "math_o2" ]; then
    TASK="math"
  fi
  if [ "$DATA_NAME" = "countdown" ]; then
    TASK="countdown"
  fi
  python -m pipeline create_prompts --task "$TASK" --data-name "$DATA_NAME" --method method_ac --split all --num-hints 5
done

#GENERATE THE SFT DATA
for TASK in countdown math sql; do
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
  CUDA_VISIBLE_DEVICES=1,2 python -m pipeline generate --task "${TASK}" --data-name "${DATA_NAME}" --method method_ac --model "${GEN_MODEL}" \
  --tensor-parallel-size 2 --split sft_train --num-samples 8 --sample-strategy random_correct --async 

  CUDA_VISIBLE_DEVICES=1,2 python -m pipeline generate --task "${TASK}" --data-name "${DATA_NAME}" --method method_ac --model "${GEN_MODEL}" \
  --tensor-parallel-size 2 --split sft_val --num-samples 8 --sample-strategy random_correct --async 
done

#TRAIN SFT AND RL MODELS FOR GENERATOR/SOLVER
for TASK in countdown math; do
  for MODEL in qwen2.5-1.5b qwen2.5-3b qwen3-4b-base; do

    python -m pipeline train_sft \
      --task "${TASK}" \
      --method method_ac \
      --run-id ${MODEL} \
      --base-model artifacts/${TASK}/models/baseline_models/${MODEL}/model \
      --dataset artifacts/${TASK}/sft_datasets/sft_train__method_ac_v3.json


    python -m pipeline train_rl \
      --task "${TASK}" \
      --method method_ac \
      --run-id ${MODEL} \
      --sft-model artifacts/${TASK}/models/method_ac_sft/${MODEL}/model \
      --train-prompts artifacts/${TASK}/problems_with_format_v3/rl_train__method_ac.parquet \
      --val-prompts artifacts/${TASK}/problems_with_format_v3/rl_val__method_ac.parquet \
      --overwrite
  
  done
done

## RUN 10 GENERATIONS / datapoint using the SOLVER