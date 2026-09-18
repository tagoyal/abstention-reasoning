#python -m pipeline generate --task countdown --method method_ac --model Qwen/Qwen3-14B --split sft_train --num-samples 8 --sample-strategy random_correct --async


for TASK in countdown competition_math; do
  for MODEL in qwen2.5-1.5b qwen2.5-3b qwen3-4b-base; do

    python -m pipeline train_sft \
      --task "${TASK}" \
      --method method_ac \
      --run-id ${MODEL} \
      --base-model artifacts/${TASK}/models/baseline_models/${MODEL}/model \
      --dataset artifacts/${TASK}/sft_datasets/sft_train__method_ac.json


    python -m pipeline train_rl \
      --task "${TASK}" \
      --method method_ac \
      --run-id ${MODEL} \
      --sft-model artifacts/${TASK}/models/method_ac_sft/${MODEL}/model \
      --train-prompts artifacts/${TASK}/problems_with_format/rl_train__method_ac.parquet \
      --val-prompts artifacts/${TASK}/problems_with_format/rl_val__method_ac.parquet \
      --overwrite
  
  done
done