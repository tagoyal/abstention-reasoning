#python -m pipeline generate --task countdown --method method_ac --model Qwen/Qwen3-14B --split sft_train --num-samples 8 --sample-strategy random_correct --async

python -m pipeline train_sft \
  --task countdown \
  --method method_ac \
  --run-id qwen2.5-1.5b \
  --base-model artifacts/countdown/models/baseline_models/qwen2.5-1.5b/model \
  --dataset artifacts/countdown/sft_datasets/sft_train__method_ac.json


python -m pipeline train_rl \
  --task countdown \
  --method method_ac \
  --run-id qwen2.5-1.5b \
  --sft-model artifacts/countdown/models/method_ac_sft/qwen2.5-1.5b/model \
  --train-prompts artifacts/countdown/problems_with_format/rl_train__method_ac.parquet \
  --val-prompts artifacts/countdown/problems_with_format/rl_val__method_ac.parquet \
  --overwrite