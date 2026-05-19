#!/bin/bash
#SBATCH --job-name=traj-eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/017264195/trajectory-prediction}"
cd "${PROJECT_DIR}"

mkdir -p logs
source .venv/bin/activate

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=/path/to/best_model.pt}"
INFOS="${INFOS:?Set INFOS=data/infos_val.pkl or data/infos_test.pkl}"
METRICS_OUT="${METRICS_OUT:-results/experiments/eval_${SLURM_JOB_ID}.json}"

PAST_LEN="${PAST_LEN:-10}"
FUTURE_LEN="${FUTURE_LEN:-20}"
MAX_WINDOWS="${MAX_WINDOWS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-512}"

echo "Evaluating checkpoint"
echo "CHECKPOINT=${CHECKPOINT}"
echo "INFOS=${INFOS}"

python code/evaluation/evaluate_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --infos "${INFOS}" \
  --past-len "${PAST_LEN}" \
  --future-len "${FUTURE_LEN}" \
  --max-windows "${MAX_WINDOWS}" \
  --batch-size "${BATCH_SIZE}" \
  --metrics-out "${METRICS_OUT}"

echo "Evaluation complete: ${METRICS_OUT}"
