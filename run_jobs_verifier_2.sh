#!/usr/bin/env bash
set -euo pipefail

MODEL="qwen2.5-3b"
TASK="competition_math"

python -m pipeline generate \
  --task "${TASK}" \
  --method method_ac \
  --model rl \
  --run-id "${MODEL}" \
  --split sft_train \
  --num-samples 10 \
  --sample-strategy random \
  --async \
  --output "artifacts/${TASK}/sft_datasets/sft_train__method_ac__verification-10s__${MODEL}.internal.json"

python -m pipeline generate \
  --task "${TASK}" \
  --method method_ac \
  --model rl \
  --run-id "${MODEL}" \
  --split sft_val \
  --num-samples 10 \
  --sample-strategy random \
  --async \
  --output "artifacts/${TASK}/sft_datasets/sft_val__method_ac__verification-10s__${MODEL}.internal.json"

python -m pipeline create_verification_data \
  --task "${TASK}" \
  --method method_a \
  --generations "artifacts/${TASK}/sft_datasets/sft_train__method_ac__verification-10s__${MODEL}.internal.json" \
  --output "artifacts/${TASK}/sft_datasets/sft_train__method_a__qh_predictions__${MODEL}.json"

python -m pipeline create_verification_data \
  --task "${TASK}" \
  --method method_a \
  --generations "artifacts/${TASK}/sft_datasets/sft_val__method_ac__verification-10s__${MODEL}.internal.json" \
  --output "artifacts/${TASK}/sft_datasets/sft_val__method_a__qh_predictions__${MODEL}.json"


python -m pipeline train_sft \
  --task "${TASK}" \
  --method method_a \
  --run-id "${MODEL}" \
  --base-model "artifacts/${TASK}/models/method_ac_models/${MODEL}/model" \
  --dataset "artifacts/${TASK}/sft_datasets/sft_train__method_a__qh_predictions__${MODEL}.json" \
  --output "artifacts/${TASK}/models/method_a_predictors/${MODEL}/model" \
  --completion-only-loss


python -m pipeline create_prompts \
  --task "${TASK}" \
  --method method_a \
  --split eval \
  --num-hints 5 \
  --no-assistant-prefix

python -m pipeline evaluate \
  --task "${TASK}" \
  --method method_ac \
  --model rl \
  --run-id "${MODEL}" \
  --prompts "artifacts/${TASK}/problems_with_format/eval__method_ac.json" \
  --output "artifacts/${TASK}/models/method_ac_models/${MODEL}/evals/eval__all-hint-levels.internal.json" \
  --num-samples 32 \
  --async

python -m pipeline evaluate \
  --task "${TASK}" \
  --method method_a \
  --model "artifacts/${TASK}/models/method_a_predictors/${MODEL}/model" \
  --prompts "artifacts/${TASK}/problems_with_format/eval__method_a.json" \
  --output "artifacts/${TASK}/models/method_a_predictors/${MODEL}/evals/eval__verifier-decisions.internal.json" \
  --max-new-tokens 16 \
  --temperature 1.0 \
  --num-samples 32 \
  --async

python -m pipeline combine_verifier_eval \
  --task "${TASK}" \
  --solver-results "artifacts/${TASK}/models/method_ac_models/${MODEL}/evals/eval__all-hint-levels.internal.json" \
  --verifier-results "artifacts/${TASK}/models/method_a_predictors/${MODEL}/evals/eval__verifier-decisions.internal.json" \
  --output "artifacts/${TASK}/models/method_a_predictors/${MODEL}/evals/eval.json" \
  --max-hints 5