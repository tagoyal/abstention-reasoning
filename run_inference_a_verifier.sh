

set -euo pipefail

TASK="$1"
MODEL="$2"
RUN_ID="${MODEL}_v3"

python chain_method_a.py \
        --task "${TASK}" \
        --eval-dataset artifacts/${TASK}/problems/eval.json \
        --verifier-model artifacts/${TASK}/models/method_a_predictors/${RUN_ID}/model \
        --solver-model artifacts/${TASK}/models/method_ac_models/${RUN_ID}/model \
        --num-samples 32 \
        --max-hints 5 \
        --async \
        --output artifacts/${TASK}/models/method_a_predictors/${RUN_ID}/evals/eval__chained_32s.json