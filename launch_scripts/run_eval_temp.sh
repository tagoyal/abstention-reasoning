#!/bin/bash
# Usage: bash launch_scripts/run_eval_temp.sh [gpus]
# Evaluate both checkpoints of the completed, unevaluated RL runs audited on 11/26.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
MODELS_ROOT="${MODELS_ROOT:-/data/tanyagoyal/models}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
GPUS="${1:-1}"
SUCCEEDED=0
FAILED=0
SKIPPED=0

run_eval() {
  local data_name="$1" method="$2" run_id="$3" compose_k="${4:-}"
  local task="${data_name}" suffix="" max_new_tokens=2048
  local run_dir="${MODELS_ROOT}/${data_name}/${method}_rl/${run_id}"
  local checkpoint model output status
  local compose_args=()
  if [[ "${data_name}" == math_o3 ]]; then
    task=math
  fi
  if [[ -n "${compose_k}" ]]; then
    suffix="__k${compose_k}"
    max_new_tokens=$((2048 * compose_k))
    compose_args=(--compose-k "${compose_k}")
  fi
  local prompts="${DATA_ROOT}/${data_name}/problems_with_format/eval__${method}${suffix}.json"

  for checkpoint in last best; do
    model="${run_dir}/${checkpoint}"
    output="${run_dir}/evals/eval${compose_k:+_k${compose_k}}_${checkpoint}_10s.json"
    if [[ ! -s "${prompts}" ]]; then
      printf 'SKIP %s/%s: missing eval prompts %s\n' "${run_dir}" "${checkpoint}" "${prompts}" >&2
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    if [[ ! -s "${model}/config.json" ]]; then
      printf 'SKIP %s: missing checkpoint config\n' "${model}" >&2
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    printf '\nEvaluating %s -> %s\n' "${model}" "${output}"
    if python -m pipeline evaluate \
      --task "${task}" \
      --data-name "${data_name}" \
      --method "${method}" \
      --model "${model}" \
      --prompts "${prompts}" \
      --output "${output}" \
      --num-samples 10 \
      --max-new-tokens "${max_new_tokens}" \
      --data-parallel-size "${GPUS}" \
      --async \
      "${compose_args[@]}"; then
      SUCCEEDED=$((SUCCEEDED + 1))
    else
      status=$?
      printf 'FAILED %s (exit %s); continuing\n' "${model}" "${status}" >&2
      FAILED=$((FAILED + 1))
    fi
  done
}

run_eval countdown baseline_composed 11_19-baseline_composed-qwen3-4b-k2 2
run_eval countdown baseline_composed 11_25-baseline_composed-qwen2.5-3b-k2-stitched 2
run_eval countdown baseline 11_17-baseline-qwen2.5-3b
run_eval countdown baseline 11_17-baseline-qwen3-4b
run_eval countdown method_ac 11_18-method_ac-qwen2.5-3b
run_eval countdown method_ac 11_18-method_ac-qwen3-4b
run_eval countdown method_b qwen2.5-1.5b_lin-a0.1
run_eval countdown method_b qwen2.5-3b_lin-a0.1
run_eval countdown method_b qwen3-4b_lin-a0.1

run_eval math_o3 baseline_composed 11_18-baseline_composed-qwen2.5-3b-k2 2
run_eval math_o3 baseline_composed 11_19-baseline_composed-qwen3-4b-k2 2
run_eval math_o3 baseline 11_17-baseline-qwen2.5-3b
run_eval math_o3 baseline 11_17-baseline-qwen3-4b
run_eval math_o3 baseline qwen2.5-1.5b
run_eval math_o3 method_ac 11_10-method_ac-qwen2.5-3b
run_eval math_o3 method_ac 11_10-method_ac-qwen3-4b
run_eval math_o3 method_ac 11_18-method_ac-qwen2.5-3b
run_eval math_o3 method_ac 11_18-method_ac-qwen3-4b
run_eval math_o3 method_b_composed 11_25-method_b_composed-qwen2.5-3b-k2_quad-a1.5_b0.1 2
run_eval math_o3 method_b 10_21-method_b-qwen2.5-1.5b_quad-a1.5_b0
run_eval math_o3 method_b 10_21-method_b-qwen2.5-1.5b_quad-a1.5_b0.05
run_eval math_o3 method_b 10_21-method_b-qwen2.5-1.5b_quad-a2_b0
run_eval math_o3 method_b 10_21-method_b-qwen2.5-1.5b_quad-a2_b0.05
run_eval math_o3 method_b 10_21-method_b-qwen2.5-3b_quad-a1.5_b0
run_eval math_o3 method_b 10_21-method_b-qwen2.5-3b_quad-a1.5_b0.05
run_eval math_o3 method_b 10_21-method_b-qwen2.5-3b_quad-a2_b0
run_eval math_o3 method_b 10_21-method_b-qwen2.5-3b_quad-a2_b0.05
run_eval math_o3 method_b 10_21-method_b-qwen3-4b_quad-a1.5_b0
run_eval math_o3 method_b 10_21-method_b-qwen3-4b_quad-a1.5_b0.05
run_eval math_o3 method_b 10_21-method_b-qwen3-4b_quad-a2_b0
run_eval math_o3 method_b 10_21-method_b-qwen3-4b_quad-a2_b0.05
run_eval math_o3 method_b qwen2.5-1.5b_quad-a2_b0.1
run_eval math_o3 method_b qwen2.5-3b_quad-a1.5_b0.1
run_eval math_o3 method_b qwen3-4b_quad-a1.5_b0.1

printf '\nFinished: %s succeeded, %s failed, %s skipped\n' "${SUCCEEDED}" "${FAILED}" "${SKIPPED}"
