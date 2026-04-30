#!/bin/bash
#SBATCH --job-name=traj-friday
#SBATCH --output=logs/friday_pipeline_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/017264195/trajectory-prediction}"
cd "${PROJECT_DIR}"

mkdir -p logs
chmod +x scripts/run_friday_pipeline.sh

# Defaults for CMPE249 Waymo setup.
export TRAIN_INFOS="${TRAIN_INFOS:-data/infos_train.pkl}"
export VAL_INFOS="${VAL_INFOS:-data/infos_val.pkl}"
export TEST_INFOS="${TEST_INFOS:-data/infos_test.pkl}"

export PAST_LEN="${PAST_LEN:-10}"
export FUTURE_LEN="${FUTURE_LEN:-20}"
export MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-2500}"
export MAX_VAL_WINDOWS="${MAX_VAL_WINDOWS:-800}"
export MAX_TEST_WINDOWS="${MAX_TEST_WINDOWS:-800}"
export MAX_NEIGHBORS="${MAX_NEIGHBORS:-12}"

export EPOCHS="${EPOCHS:-12}"
export LR="${LR:-0.001}"
export SEED="${SEED:-42}"

export BATCH_SIZE_LSTM="${BATCH_SIZE_LSTM:-64}"
export BATCH_SIZE_TRANSFORMER="${BATCH_SIZE_TRANSFORMER:-64}"
export BATCH_SIZE_MULTI="${BATCH_SIZE_MULTI:-48}"

export RUN_TAG="${RUN_TAG:-friday_${SLURM_JOB_ID}}"
export BASE_DIR="${BASE_DIR:-results/experiments/${RUN_TAG}}"

echo "Submitting Friday pipeline with RUN_TAG=${RUN_TAG}"
bash scripts/run_friday_pipeline.sh
