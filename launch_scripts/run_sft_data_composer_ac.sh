#!/bin/bash
# Fixed-prompt method_ac_composed SFT data from existing unaided baseline traces.
# Usage: bash launch_scripts/run_sft_data_composer_ac.sh <task> [compose_k] [data_name]
# Defaults: K=2; data_name=countdown or math_o3.
# BASELINE_SFT_DIR overrides the directory containing sft_{train,val}__baseline.json.

set -euo pipefail

TASK="$1"
COMPOSE_K="${2:-${COMPOSE_K:-2}}"
case "${TASK}" in
  math) DEFAULT_DATA_NAME="math_o3" ;;
  countdown) DEFAULT_DATA_NAME="countdown" ;;
  *) echo "method_ac_composed supports only math and countdown." >&2; exit 1 ;;
esac
DATA_NAME="${3:-${DEFAULT_DATA_NAME}}"
METHOD="method_ac_composed"
BASELINE_SFT_DIR="${BASELINE_SFT_DIR:-data/${DATA_NAME}/sft_datasets}"

for SPLIT in sft_train sft_val; do
  SOURCE="${BASELINE_SFT_DIR}/${SPLIT}__baseline.json"
  if [[ ! -f "${SOURCE}" ]] &&
     ! compgen -G "${BASELINE_SFT_DIR}/${SPLIT}__baseline.shard*.json" > /dev/null; then
    echo "Baseline source traces not found: ${SOURCE}" >&2
    exit 1
  fi
done

for SPLIT in sft_train sft_val; do
  PROMPTS="data/${DATA_NAME}/problems_with_format/${SPLIT}__${METHOD}__k${COMPOSE_K}.json"
  if [[ ! -f "${PROMPTS}" ]] &&
     ! compgen -G "data/${DATA_NAME}/problems_with_format/${SPLIT}__${METHOD}__k${COMPOSE_K}.shard*.json" > /dev/null; then
    python -m pipeline create_prompts --task "${TASK}" --method "${METHOD}" \
      --compose-k "${COMPOSE_K}" --split "${SPLIT}" --data-name "${DATA_NAME}"
  fi
  python -m pipeline stitch_composed_sft --task "${TASK}" --method "${METHOD}" \
    --compose-k "${COMPOSE_K}" --split "${SPLIT}" \
    --source "${BASELINE_SFT_DIR}/${SPLIT}__baseline.json" --data-name "${DATA_NAME}"
done
