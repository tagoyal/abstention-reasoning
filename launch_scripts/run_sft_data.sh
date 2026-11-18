#!/bin/bash
# Usage: bash launch_scripts/run_sft_data.sh <task> [gen_model] [compose_k]
# Launched per task via submit_sft_data_template.yaml.
#
# MODE=composed (default): SFT data for baseline_composed (K problems per datapoint, math/countdown only).
# MODE=hints: the original method_b (hint mix) + method_ac SFT data.
set -euo pipefail

TASK="$1"
GEN_MODEL="${2:-Qwen/Qwen3-14B}"
COMPOSE_K="${3:-${COMPOSE_K:-2}}"
MODE="${MODE:-composed}"
# Exact share of examples per hint count. Each problem is capped at its count.
HINT_MIX="${HINT_MIX:-0:0.4,1:0.3,2:0.3}"
# Suffix for the output files: <split>__method_b__<SUFFIX>.json. Train on them with --dataset.
SUFFIX="${SUFFIX:-hintmix}"
# Most hints any problem can get; must match the keys in HINT_MIX.
MAX_HINTS="${MAX_HINTS:-2}"

DATA_NAME="$TASK"
if [ "$TASK" = "sql" ]; then
  DATA_NAME="sql_2"
fi
if [ "$TASK" = "math" ]; then
  DATA_NAME="math_o3"
fi

if [ "$MODE" = "composed" ]; then
  if [ "$TASK" != "math" ] && [ "$TASK" != "countdown" ]; then
    echo "baseline_composed supports only math and countdown (got: $TASK)" >&2
    exit 1
  fi
  # Baseline uses the CLI default of 2048 (generate --max-new-tokens, train_rl --max-response-length); scale by K.
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$((2048 * COMPOSE_K))}"
  NUM_SAMPLES="${NUM_SAMPLES:-8}"

  # Build prompts if missing. Partitions are untouched; groups are fixed by the seed.
  if [ ! -f "data/${DATA_NAME}/problems_with_format/sft_train__baseline_composed__k${COMPOSE_K}.json" ] || \
     [ ! -f "data/${DATA_NAME}/problems_with_format/sft_val__baseline_composed__k${COMPOSE_K}.json" ]; then
    python -m pipeline create_prompts --task "${TASK}" --method baseline_composed \
      --compose-k "${COMPOSE_K}" --split all --data-name "${DATA_NAME}"
  fi

  # Outputs: data/${DATA_NAME}/sft_datasets/{split}__baseline_composed__k${COMPOSE_K}.json
  # A sample counts as correct only if all K subparts are correct.
  for SPLIT in sft_train sft_val; do
    python -m pipeline generate --task "${TASK}" --method baseline_composed \
      --compose-k "${COMPOSE_K}" --async \
      --model "${GEN_MODEL}" --split "${SPLIT}" \
      --num-samples "${NUM_SAMPLES}" --sample-strategy random_correct \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --data-name "${DATA_NAME}"
  done
elif [ "$MODE" = "hints" ]; then
  for SPLIT in sft_train sft_val; do
    # generate resumes from existing outputs, so remove them to regenerate from scratch
    rm -f "data/${DATA_NAME}/.scratch/${SPLIT}__baseline__probe.json" \
          "data/${DATA_NAME}/sft_datasets/${SPLIT}__method_b__${SUFFIX}.json"

    python -m pipeline generate --task "${TASK}" --method baseline --async \
      --model "${GEN_MODEL}" --split "${SPLIT}" --no-hints --num-samples 8 \
      --data-name "${DATA_NAME}" \
      --output data/${DATA_NAME}/.scratch/${SPLIT}__baseline__probe.json

    # Schedule hints from that probe -- harder problems get more -- and oversample
    # until the worst hint bucket is half correct. The profile is derived in memory; only the
    # dataset is written.
    python -m pipeline generate --task "${TASK}" --method method_b --async \
        --model "${GEN_MODEL}" --split "${SPLIT}" --max-hints "${MAX_HINTS}" \
        --hint-schedule data/${DATA_NAME}/.scratch/${SPLIT}__baseline__probe.json \
        --hint-mix "${HINT_MIX}" --target-correct-rate 0.5 \
        --output "data/${DATA_NAME}/sft_datasets/${SPLIT}__method_b__${SUFFIX}.json" \
        --data-name "${DATA_NAME}"
  done

  python -m pipeline generate --task "${TASK}" --data-name "${DATA_NAME}" --method method_ac --model "${GEN_MODEL}" \
  --split sft_train --num-samples 8 --sample-strategy random_correct --async 

  python -m pipeline generate --task "${TASK}" --data-name "${DATA_NAME}" --method method_ac --model "${GEN_MODEL}" \
  --split sft_val --num-samples 8 --sample-strategy random_correct --async 
else
  echo "Unknown MODE: $MODE (expected composed or hints)" >&2
  exit 1
fi
