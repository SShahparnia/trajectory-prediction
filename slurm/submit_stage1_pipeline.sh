#!/bin/bash
set -euo pipefail

# End-to-end Stage 1 pipeline submission:
# 1) Train on train split with explicit val split
# 2) Evaluate best checkpoint on val split
# 3) Evaluate best checkpoint on test split

PROJECT_DIR="${PROJECT_DIR:-/home/017264195/trajectory-prediction}"
cd "${PROJECT_DIR}"

mkdir -p logs results/experiments

DEFAULTS_FILE="${DEFAULTS_FILE:-slurm/pipeline_defaults.env}"
if [[ -f "${DEFAULTS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${DEFAULTS_FILE}"
fi

TRAIN_INFOS="${TRAIN_INFOS:-data/infos_train.pkl}"
VAL_INFOS="${VAL_INFOS:-data/infos_val.pkl}"
TEST_INFOS="${TEST_INFOS:-data/infos_test.pkl}"

RUN_TAG="${RUN_TAG:-stage1_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/experiments/${RUN_TAG}}"

PAST_LEN="${PAST_LEN:-10}"
FUTURE_LEN="${FUTURE_LEN:-20}"
MAX_WINDOWS="${MAX_WINDOWS:-120000}"
VAL_MAX_WINDOWS="${VAL_MAX_WINDOWS:-30000}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
SEED="${SEED:-42}"

if [[ ! -f "${TRAIN_INFOS}" ]]; then
  echo "ERROR: TRAIN_INFOS not found: ${TRAIN_INFOS}" >&2
  exit 1
fi
if [[ ! -f "${VAL_INFOS}" ]]; then
  echo "ERROR: VAL_INFOS not found: ${VAL_INFOS}" >&2
  exit 1
fi
if [[ ! -f "${TEST_INFOS}" ]]; then
  echo "WARN: TEST_INFOS not found: ${TEST_INFOS}" >&2
  echo "WARN: Falling back TEST_INFOS to VAL_INFOS for this run." >&2
  TEST_INFOS="${VAL_INFOS}"
fi

echo "Submitting Stage 1 train job..."
TRAIN_JOB_ID=$(
  sbatch --parsable \
    --export=ALL,DEFAULTS_FILE="${DEFAULTS_FILE}",PROJECT_DIR="${PROJECT_DIR}",TRAIN_INFOS="${TRAIN_INFOS}",VAL_INFOS="${VAL_INFOS}",OUT_DIR="${OUT_DIR}",PAST_LEN="${PAST_LEN}",FUTURE_LEN="${FUTURE_LEN}",MAX_WINDOWS="${MAX_WINDOWS}",VAL_MAX_WINDOWS="${VAL_MAX_WINDOWS}",EPOCHS="${EPOCHS}",BATCH_SIZE="${BATCH_SIZE}",LR="${LR}",HIDDEN_DIM="${HIDDEN_DIM}",SEED="${SEED}" \
    slurm/train_stage1_lstm.sbatch
)
echo "TRAIN_JOB_ID=${TRAIN_JOB_ID}"

CHECKPOINT_PATH="${OUT_DIR}/best_model.pt"

echo "Submitting validation eval job..."
VAL_JOB_ID=$(
  sbatch --parsable \
    --dependency=afterok:${TRAIN_JOB_ID} \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CHECKPOINT="${CHECKPOINT_PATH}",INFOS="${VAL_INFOS}",PAST_LEN="${PAST_LEN}",FUTURE_LEN="${FUTURE_LEN}",MAX_WINDOWS="${VAL_MAX_WINDOWS}",METRICS_OUT="${OUT_DIR}/metrics_val.json" \
    slurm/eval_checkpoint.sbatch
)
echo "VAL_JOB_ID=${VAL_JOB_ID}"

echo "Submitting test eval job..."
TEST_JOB_ID=$(
  sbatch --parsable \
    --dependency=afterok:${TRAIN_JOB_ID} \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CHECKPOINT="${CHECKPOINT_PATH}",INFOS="${TEST_INFOS}",PAST_LEN="${PAST_LEN}",FUTURE_LEN="${FUTURE_LEN}",MAX_WINDOWS="${VAL_MAX_WINDOWS}",METRICS_OUT="${OUT_DIR}/metrics_test.json" \
    slurm/eval_checkpoint.sbatch
)
echo "TEST_JOB_ID=${TEST_JOB_ID}"

echo "Pipeline submitted."
echo "Run directory: ${OUT_DIR}"
echo "Track logs with: squeue -u \$USER"
echo "Check outputs in: ${OUT_DIR}"
