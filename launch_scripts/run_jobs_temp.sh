python -m pipeline train_classifier \
  --task countdown --method baseline --run-id qwen2.5-1.5b \
  --base-model /data/tanyagoyal/models/countdown/baseline_rl/qwen2.5-1.5b/model \
  --use-lora \
  --epochs 3 \
  --gradient-accumulation-steps 4 \
  --max-length 2048 \
  --depth-eval-max-new-tokens 1
