#!/bin/bash
set -euo pipefail

# End-to-end Friday milestone:
# 1) Train LSTM, Transformer, Multi-agent LSTM
# 2) Evaluate each on test split
# 3) Save qualitative plots per model
# 4) Save comparison charts and summary CSV

PROJECT_DIR="${PROJECT_DIR:-/home/017264195/trajectory-prediction}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found in ${PROJECT_DIR}" >&2
  exit 1
fi
source .venv/bin/activate

mkdir -p logs results/experiments

TRAIN_INFOS="${TRAIN_INFOS:-data/infos_train.pkl}"
VAL_INFOS="${VAL_INFOS:-data/infos_val.pkl}"
TEST_INFOS="${TEST_INFOS:-data/infos_test.pkl}"

PAST_LEN="${PAST_LEN:-10}"
FUTURE_LEN="${FUTURE_LEN:-20}"
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-2500}"
MAX_VAL_WINDOWS="${MAX_VAL_WINDOWS:-800}"
MAX_TEST_WINDOWS="${MAX_TEST_WINDOWS:-800}"
MAX_NEIGHBORS="${MAX_NEIGHBORS:-12}"

EPOCHS="${EPOCHS:-12}"
LR="${LR:-0.001}"
SEED="${SEED:-42}"

BATCH_SIZE_LSTM="${BATCH_SIZE_LSTM:-64}"
BATCH_SIZE_TRANSFORMER="${BATCH_SIZE_TRANSFORMER:-64}"
BATCH_SIZE_MULTI="${BATCH_SIZE_MULTI:-48}"

RUN_TAG="${RUN_TAG:-friday_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-results/experiments/${RUN_TAG}}"
LSTM_DIR="${BASE_DIR}/lstm"
TRANSFORMER_DIR="${BASE_DIR}/transformer"
MULTI_DIR="${BASE_DIR}/multi_lstm"
COMPARE_DIR="${BASE_DIR}/comparison"

for f in "${TRAIN_INFOS}" "${VAL_INFOS}" "${TEST_INFOS}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: Missing infos file: ${f}" >&2
    exit 1
  fi
done

echo "Run tag: ${RUN_TAG}"
echo "Output base: ${BASE_DIR}"
echo "Python: $(python --version 2>&1)"

python train/train_trajectory_model.py \
  --model-type lstm \
  --train-infos "${TRAIN_INFOS}" \
  --val-infos "${VAL_INFOS}" \
  --past-len "${PAST_LEN}" \
  --future-len "${FUTURE_LEN}" \
  --max-train-windows "${MAX_TRAIN_WINDOWS}" \
  --max-val-windows "${MAX_VAL_WINDOWS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE_LSTM}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --out-dir "${LSTM_DIR}"

python train/train_trajectory_model.py \
  --model-type transformer \
  --train-infos "${TRAIN_INFOS}" \
  --val-infos "${VAL_INFOS}" \
  --past-len "${PAST_LEN}" \
  --future-len "${FUTURE_LEN}" \
  --max-train-windows "${MAX_TRAIN_WINDOWS}" \
  --max-val-windows "${MAX_VAL_WINDOWS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE_TRANSFORMER}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --out-dir "${TRANSFORMER_DIR}"

python train/train_trajectory_model.py \
  --model-type multi_lstm \
  --train-infos "${TRAIN_INFOS}" \
  --val-infos "${VAL_INFOS}" \
  --past-len "${PAST_LEN}" \
  --future-len "${FUTURE_LEN}" \
  --max-neighbors "${MAX_NEIGHBORS}" \
  --max-train-windows "${MAX_TRAIN_WINDOWS}" \
  --max-val-windows "${MAX_VAL_WINDOWS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE_MULTI}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --out-dir "${MULTI_DIR}"

python code/evaluation/evaluate_trajectory_checkpoint.py \
  --checkpoint "${LSTM_DIR}/best_model.pt" \
  --infos "${TEST_INFOS}" \
  --past-len "${PAST_LEN}" \
  --future-len "${FUTURE_LEN}" \
  --max-windows "${MAX_TEST_WINDOWS}" \
  --metrics-out "${LSTM_DIR}/test_metrics.json" \
  --viz-out-dir "${LSTM_DIR}/qualitative_test" \
  --num-viz 12

python code/evaluation/evaluate_trajectory_checkpoint.py \
  --checkpoint "${TRANSFORMER_DIR}/best_model.pt" \
  --infos "${TEST_INFOS}" \
  --past-len "${PAST_LEN}" \
  --future-len "${FUTURE_LEN}" \
  --max-windows "${MAX_TEST_WINDOWS}" \
  --metrics-out "${TRANSFORMER_DIR}/test_metrics.json" \
  --viz-out-dir "${TRANSFORMER_DIR}/qualitative_test" \
  --num-viz 12

python code/evaluation/evaluate_trajectory_checkpoint.py \
  --checkpoint "${MULTI_DIR}/best_model.pt" \
  --infos "${TEST_INFOS}" \
  --past-len "${PAST_LEN}" \
  --future-len "${FUTURE_LEN}" \
  --max-neighbors "${MAX_NEIGHBORS}" \
  --max-windows "${MAX_TEST_WINDOWS}" \
  --metrics-out "${MULTI_DIR}/test_metrics.json" \
  --viz-out-dir "${MULTI_DIR}/qualitative_test" \
  --num-viz 12

python scripts/plot_model_comparison.py \
  --run lstm "${LSTM_DIR}/train_history.csv" "${LSTM_DIR}/test_metrics.json" \
  --run transformer "${TRANSFORMER_DIR}/train_history.csv" "${TRANSFORMER_DIR}/test_metrics.json" \
  --run multi_lstm "${MULTI_DIR}/train_history.csv" "${MULTI_DIR}/test_metrics.json" \
  --out-dir "${COMPARE_DIR}"

echo "Pipeline complete."
echo "Artifacts:"
echo "  ${LSTM_DIR}/qualitative_test"
echo "  ${TRANSFORMER_DIR}/qualitative_test"
echo "  ${MULTI_DIR}/qualitative_test"
echo "  ${COMPARE_DIR}"
