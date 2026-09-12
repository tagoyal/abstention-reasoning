python -m pipeline generate --task countdown --method method_ac --model Qwen/Qwen3-14B --split sft_train --num-samples 8 --sample-strategy random_correct --async

python -m pipeline generate --task countdown --method method_ac --model Qwen/Qwen3-14B --split sft_val --num-samples 8 --sample-strategy random_correct --async
