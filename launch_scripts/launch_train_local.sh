#!/bin/bash
# Run inside an initialized pod, preferably in tmux:
#   bash launch_scripts/launch_train_local.sh [extra train_rl flags...]
# Edit the arrays below just as in launch_train_jobs.sh. Pull code before
# starting the batch, not while its jobs are running.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TASKS=(math)
DATA_NAMES=(math_o3)
MODELS=(qwen2.5-1.5b qwen2.5-3b qwen3-4b)
# Use GPU indices shown by nvidia-smi inside this pod.
GPU_POOL=(0 1 2 3)
ALPHA=1.5
BASE=0
TRAINING_FLAGS=(--interleave-rl)

POLL_SECONDS=5
MAX_USED_MEMORY_MIB=512
LOG_ROOT="$SCRIPT_DIR/../logs/local_train"

declare -A MODEL_MAP=(
  ["qwen2.5-1.5b"]="Qwen/Qwen2.5-1.5B"
  ["qwen2.5-3b"]="Qwen/Qwen2.5-3B"
  ["qwen3-4b"]="Qwen/Qwen3-4B-Base"
  ["qwen3-8b"]="Qwen/Qwen3-8B-Base"
)
declare -A GPU_MAP=(
  ["qwen2.5-1.5b"]=1
  ["qwen2.5-3b"]=1
  ["qwen3-4b"]=1
  ["qwen3-8b"]=2
)

JOB_ARGS=()
for TASK in "${TASKS[@]}"; do
  for DATA_NAME in "${DATA_NAMES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      if [[ -z "${MODEL_MAP[$MODEL]:-}" || -z "${GPU_MAP[$MODEL]:-}" ]]; then
        printf 'Unknown model: %s\n' "$MODEL" >&2
        exit 1
      fi
      JOB_ARGS+=(--job "$TASK" "$MODEL" "${MODEL_MAP[$MODEL]}" "$DATA_NAME"
        "$ALPHA" "$BASE" "${GPU_MAP[$MODEL]}")
    done
  done
done

exec python "$SCRIPT_DIR/local_train_scheduler.py" \
  --gpus "${GPU_POOL[@]}" \
  --log-root "$LOG_ROOT" \
  --poll-seconds "$POLL_SECONDS" \
  --max-used-memory-mib "$MAX_USED_MEMORY_MIB" \
  "${JOB_ARGS[@]}" -- "${TRAINING_FLAGS[@]}" "$@"
