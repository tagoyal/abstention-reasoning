#!/bin/bash
# Submits one k8s job per TASK, each running `run_jobs_temp.sh <task>
# <data_name>` inside the pod -- which itself cycles through every run-id
# (qwen2.5-1.5b/3b, qwen3-4b/8b), training a LoRA depth-accuracy classifier
# on top of each run-id's baseline_rl model
# (models/<data_name>/baseline_rl/<run_id>/model). Generates the pod
# manifest in-memory (via sed substitution on submit_classifier_template.yaml,
# which uses __TASK__/__DATA_NAME__/__GPUS__ placeholders) and pipes it
# directly to `kubectl create -f -` -- no per-combo yaml/sh files are
# written to disk or need to be committed. Only run_jobs_temp.sh (referenced
# inside the pod, which git-clones the repo) must be pushed beforehand.
#
# Usage: bash launch_classifier_jobs.sh

set -euo pipefail

TASKS=(math countdown sql)

declare -A DATA_NAME_MAP=(
  ["math"]="math_o1"
  ["countdown"]="countdown"
  ["sql"]="sql_conceptual"
)

# Classifier training (LoRA, single run-id at a time inside run_jobs_temp.sh)
# fits every model size on a single GPU.
GPUS=1

for TASK in "${TASKS[@]}"; do
  DATA_NAME="${DATA_NAME_MAP[$TASK]}"
  sed "s|__TASK__|${TASK}|; s|__DATA_NAME__|${DATA_NAME}|; s|__GPUS__|${GPUS}|" launch_scripts/submit_classifier_template.yaml \
    | kubectl create -f -
done
