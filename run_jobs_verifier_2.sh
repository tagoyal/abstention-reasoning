#!/usr/bin/env bash
set -euo pipefail

MODEL="qwen2.5-1.5b"
TASK="countdown"

python -m pipeline evaluate \
  --task "${TASK}" \
  --method method_ac \
  --model rl \
  --run-id "${MODEL}" \
  --prompts "artifacts/${TASK}/problems_with_format/eval__method_ac.json" \
  --output "artifacts/${TASK}/models/method_ac_models/${MODEL}/evals/eval__all-hint-levels.internal.json" \
  --num-samples 32 \
  --async

