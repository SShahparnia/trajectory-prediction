#!/bin/bash
#SBATCH --job-name=traj-s1-lstm
#SBATCH --output=logs/stage1_train_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULTS_FILE="${DEFAULTS_FILE:-${SCRIPT_DIR}/pipeline_defaults.env}"
if [[ -f "${DEFAULTS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${DEFAULTS_FILE}"
fi

PROJECT_DIR="${PROJECT_DIR:-/home/017264195/trajectory-prediction}"
cd "${PROJECT_DIR}"

mkdir -p logs
source .venv/bin/activate

TRAIN_INFOS="${TRAIN_INFOS:-data/infos_train.pkl}"
VAL_INFOS="${VAL_INFOS:-data/infos_val.pkl}"
OUT_DIR="${OUT_DIR:-results/experiments/stage1_lstm_${SLURM_JOB_ID}}"

PAST_LEN="${PAST_LEN:-10}"
FUTURE_LEN="${FUTURE_LEN:-20}"
MAX_WINDOWS="${MAX_WINDOWS:-120000}"
VAL_MAX_WINDOWS="${VAL_MAX_WINDOWS:-30000}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
SEED="${SEED:-42}"

echo "Starting stage1 training"
echo "TRAIN_INFOS=${TRAIN_INFOS}"
echo "VAL_INFOS=${VAL_INFOS}"
echo "OUT_DIR=${OUT_DIR}"

python train/train_lstm_baseline.py \
  --infos "${TRAIN_INFOS}" \
  --val-infos "${VAL_INFOS}" \
  --past-len "${PAST_LEN}" \
  --future-len "${FUTURE_LEN}" \
  --max-windows "${MAX_WINDOWS}" \
  --val-max-windows "${VAL_MAX_WINDOWS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --seed "${SEED}" \
  --out-dir "${OUT_DIR}"

echo "Training complete: ${OUT_DIR}"
