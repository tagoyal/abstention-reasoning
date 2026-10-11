#!/bin/bash
# Submits one k8s job per (TASK, MODEL) combo, each running
# `run_train.sh <task> <model> <hf_model>` inside the pod (train_sft then
# train_rl, baseline method, math_o1 data). Generates the pod manifest
# in-memory (via sed substitution on submit_train_template.yaml, which uses
# __TASK__/__MODEL__/__HF_MODEL__ placeholders) and pipes it directly to
# `kubectl create -f -` -- no per-combo yaml/sh files are written to disk or
# need to be committed. Only run_train.sh (referenced inside the pod, which
# git-clones the repo) must be pushed beforehand.
#
# Usage: bash launch_train_jobs.sh



set -euo pipefail

# TASKS=(math)
# MODELS=(qwen2.5-3b qwen3-4b-base)

# for TASK in "${TASKS[@]}"; do
#   for MODEL in "${MODELS[@]}"; do
#     sed "s|__TASK__|${TASK}|; s|__MODEL__|${MODEL}|" submit_train_template.yaml \
#       | kubectl create -f -
#   done
# done

TASKS=(math)
DATA_NAME=math_o2
MODELS=(qwen2.5-1.5b qwen2.5-3b qwen3-4b)
ALPHA=0.5
BASE=0.05


# Short name -> HuggingFace base model id. Qwen3 models use the -Base
# (non-instruct/non-reasoning) variant for cold-start SFT/RL, matching the
# convention used elsewhere in launch_scripts (e.g. run_jobs_temp.sh).
declare -A MODEL_MAP=(
  ["qwen2.5-1.5b"]="Qwen/Qwen2.5-1.5B"
  ["qwen2.5-3b"]="Qwen/Qwen2.5-3B"
  ["qwen3-4b"]="Qwen/Qwen3-4B-Base"
  ["qwen3-8b"]="Qwen/Qwen3-8B-Base"
)

# GPUs requested per model. verl's n_gpus_per_node auto-detects via
# torch.cuda.device_count(), so bumping the k8s GPU request here is enough
# to make FSDP shard params/optimizer state across GPUs -- no extra
# --n-gpus-per-node flag needed. The 8b model's unsharded Adam optimizer
# state doesn't fit on a single GPU alongside the colocated vLLM rollout
# engine (see launch_train_jobs OOM), so it gets 2 GPUs; smaller models fit
# on 1.
declare -A GPU_MAP=(
  ["qwen2.5-1.5b"]=1
  ["qwen2.5-3b"]=1
  ["qwen3-4b"]=1
  ["qwen3-8b"]=2
)


for TASK in "${TASKS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    HF_MODEL="${MODEL_MAP[$MODEL]}"
    GPUS="${GPU_MAP[$MODEL]}"
    sed "s|__TASK__|${TASK}|; s|__MODEL__|${MODEL}|; s|__HF_MODEL__|${HF_MODEL}|; s|__GPUS__|${GPUS}|; s|__DATA_NAME__|${DATA_NAME}|; s|__ALPHA__|${ALPHA}|; s|__BASE__|${BASE}|" launch_scripts/submit_train_template.yaml \
      | kubectl create -f -
  done
done
