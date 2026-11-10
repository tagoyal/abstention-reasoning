#!/bin/bash
# Re-submits the 4 (TASK, MODEL, DATA_NAME) jobs that errored out with CUDA
# OOM on slc01-cl02-hgx-0202 (pods 77w5t, w2g6m, t9v79, lwlqj). The template
# now excludes that node via nodeAffinity, so re-running the same combos
# should land on a different node.
#
# Usage: bash launch_failed_jobs.sh

set -euo pipefail

# Each row: TASK MODEL HF_MODEL GPUS DATA_NAME
JOBS=(
  "countdown qwen2.5-1.5b Qwen/Qwen2.5-1.5B 1 countdown"
  "countdown qwen2.5-3b   Qwen/Qwen2.5-3B   1 countdown"
  "sql       qwen2.5-3b   Qwen/Qwen2.5-3B   1 sql_conceptual"
  "sql       qwen3-4b     Qwen/Qwen3-4B-Base 1 sql_conceptual"
)

for JOB in "${JOBS[@]}"; do
  read -r TASK MODEL HF_MODEL GPUS DATA_NAME <<< "${JOB}"
  sed "s|__TASK__|${TASK}|; s|__MODEL__|${MODEL}|; s|__HF_MODEL__|${HF_MODEL}|; s|__GPUS__|${GPUS}|; s|__DATA_NAME__|${DATA_NAME}|" launch_scripts/submit_train_template.yaml \
    | kubectl create -f -
done
