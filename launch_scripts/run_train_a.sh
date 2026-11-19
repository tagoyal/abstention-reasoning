#!/bin/bash
# Usage: bash launch_scripts/run_train_a.sh <task> <model_short_name> <hf_base_model> <data_name>
# The HF model argument (and launcher's fifth argument) are unused.
# AC_RUN_ID selects the trained AC run; defaults to the model short name.

set -euo pipefail

if (( $# < 4 )); then
  echo "Usage: $0 <task> <model_short_name> <hf_base_model> <data_name>" >&2
  exit 2
fi

TASK="$1"
MODEL="$2"
DATA_NAME="$4"
AC_RUN_ID="${AC_RUN_ID:-${MODEL}}"
AC_MODEL="models/${DATA_NAME}/method_ac_rl/${AC_RUN_ID}/model"
NUM_HINTS="${NUM_HINTS:-5}"
NUM_SAMPLES=10

case "${TASK}" in
  countdown|math) ;;
  *)
    echo "Method A supports countdown and math (got: ${TASK})" >&2
    exit 1
    ;;
esac

if [[ ! -f "${AC_MODEL}/config.json" ]]; then
  echo "Trained Method AC model not found: ${AC_MODEL}. Set AC_RUN_ID to its run directory name." >&2
  exit 1
fi

PROMPTS_DIR="data/${DATA_NAME}/problems_with_format"
DS_DIR="data/${DATA_NAME}/sft_datasets"
for SPLIT in train val; do
  SOLVER_SPLIT="rl_ver_${SPLIT}"
  if [[ ! -f "${PROMPTS_DIR}/${SOLVER_SPLIT}__method_ac.parquet" ]] &&
     ! compgen -G "${PROMPTS_DIR}/${SOLVER_SPLIT}__method_ac.shard*.parquet" > /dev/null; then
    python -m pipeline create_prompts \
      --task "${TASK}" \
      --method method_ac \
      --data-name "${DATA_NAME}" \
      --split "${SOLVER_SPLIT}" \
      --num-hints "${NUM_HINTS}"
  fi

  # Existing solver evaluations are reused; labels are rebuilt per rollout.
  python -m pipeline create_verification_data \
    --task "${TASK}" \
    --method method_a \
    --data-name "${DATA_NAME}" \
    --run-solver \
    --solver-method method_ac \
    --solver-run-id "${AC_RUN_ID}" \
    --solver-split "${SOLVER_SPLIT}" \
    --solver-async \
    --num-samples "${NUM_SAMPLES}" \
    --output "${DS_DIR}/sft_${SPLIT}__method_a__${AC_RUN_ID}.json"
done

python -m pipeline train_sft \
  --task "${TASK}" \
  --method method_a \
  --run-id "${AC_RUN_ID}" \
  --base-model "${AC_MODEL}" \
  --data-name "${DATA_NAME}" \
  --dataset "${DS_DIR}/sft_train__method_a__${AC_RUN_ID}.json" \
  --eval-dataset "${DS_DIR}/sft_val__method_a__${AC_RUN_ID}.json" \
  --completion-only-loss
