#python -m pipeline generate --task competition_math --method method_ac --model Qwen/Qwen3-14B --split sft_train --num-samples 8 --sample-strategy random_correct --async

# python -m pipeline train_sft \
#   --task competition_math \
#   --method method_ac \
#   --run-id qwen2.5-1.5b \
#   --base-model artifacts/competition_math/models/baseline_models/qwen2.5-1.5b/model \
#   --dataset artifacts/competition_math/sft_datasets/sft_train__method_ac.json


python -m pipeline train_rl \
  --task competition_math \
  --method method_ac \
  --run-id qwen2.5-1.5b \
  --sft-model artifacts/competition_math/models/method_ac_sft/qwen2.5-1.5b/model \
  --train-prompts artifacts/competition_math/problems_with_format/rl_train__method_ac.parquet \
  --val-prompts artifacts/competition_math/problems_with_format/rl_val__method_ac.parquet  --overwrite