# Trajectory Prediction (Waymo)

Code for building trajectory windows from processed Waymo metadata, training forecasting models, and evaluating with ADE/FDE plus qualitative rollouts.

The repo is organized around a small Python package (`traj_code/`), thin entry-point scripts (`train/`, `scripts/`, `eda/`), and Slurm helpers for HPC runs. Model checkpoints and plots are written under `results/` (gitignored).

---

## Repository layout

```
trajectory-prediction/
├── traj_code/              # Core library
│   ├── data_pipeline/      # Window builders (single- and multi-agent)
│   ├── feature_engineering/
│   ├── models/             # LSTM, Transformer, multi-agent, multimodal
│   └── evaluation/         # Metrics and checkpoint evaluation
├── train/                  # Training entry points
├── eda/                    # Exploratory analysis on metadata and LiDAR
├── scripts/                # Data splits, comparisons, LiDAR/rollout visuals
├── slurm/                  # Batch jobs and pipeline submit scripts
└── data/                   # Train/val/test infos pickles (see data/README.txt)
```

---

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
```

Dependencies are listed in `requirements.txt` (PyTorch, scikit-learn, XGBoost for the optional traditional baseline, matplotlib, etc.).

### Data

Training expects processed Waymo **infos** pickles (`*_infos_*.pkl`), not raw TFRecords. This repo ships split metadata under `data/`:

- `data/infos_train.pkl`
- `data/infos_val.pkl`
- `data/infos_test.pkl`

Each entry points at LiDAR `.npy` paths on shared storage (e.g. `/scratch/.../waymo_processed_data_v0_5_0/`). You need that tree mounted to run LiDAR EDA or BEV simulations; trajectory training only needs the pickles and valid box annotations inside them.

To rebuild splits from full HPC train+val infos:

```bash
python scripts/build_expanded_data_splits.py \
  --train-infos /path/to/waymo_processed_data_v0_5_0_infos_train.pkl \
  --val-infos /path/to/waymo_processed_data_v0_5_0_infos_val.pkl
```

---

## Usage

### Exploratory analysis

Object-level stats and plots:

```bash
python eda/EDA.py \
  --infos data/infos_train.pkl \
  --max-samples 500 \
  --out-dir results/eda_train_500
```

Point-cloud EDA (requires processed `.npy` on disk):

```bash
python eda/EDA_pointcloud.py \
  --infos data/infos_train.pkl \
  --processed-root /scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0 \
  --max-samples 200 \
  --out-dir results/pointcloud_eda_train_200
```

### Train a single-agent LSTM (Stage 1)

```bash
python train/train_lstm_baseline.py \
  --infos data/infos_train.pkl \
  --val-infos data/infos_val.pkl \
  --past-len 10 --future-len 20 \
  --out-dir results/experiments/stage1_lstm
```

### Train LSTM / Transformer / multi-agent LSTM

```bash
python train/train_trajectory_model.py \
  --model-type lstm \
  --train-infos data/infos_train.pkl \
  --val-infos data/infos_val.pkl \
  --past-len 10 --future-len 20 \
  --out-dir results/experiments/lstm
```

Supported `--model-type` values: `lstm`, `transformer`, `multi_lstm`, `multi_lstm_attn`, `lstm_multimodal`.

### Evaluate a checkpoint

```bash
python traj_code/evaluation/evaluate_trajectory_checkpoint.py \
  --checkpoint results/experiments/lstm/best_model.pt \
  --infos data/infos_test.pkl \
  --past-len 10 --future-len 20 \
  --metrics-out results/experiments/lstm/test_metrics.json \
  --viz-out-dir results/experiments/lstm/qualitative_test \
  --num-viz 12
```

Single-agent LSTM checkpoints from `train_lstm_baseline.py` can use `traj_code/evaluation/evaluate_checkpoint.py` instead.

### Optional: traditional ML baselines

```bash
python train/train_traditional_models.py
```

Trains Random Forest and XGBoost on engineered features; writes under `results/train_random_forest/` and `results/train_xgboost/`.

### Comparison plots and simulations

```bash
python scripts/plot_model_comparison.py \
  --run lstm results/experiments/lstm/train_history.csv results/experiments/lstm/test_metrics.json \
  --run transformer results/experiments/transformer/train_history.csv results/experiments/transformer/test_metrics.json \
  --out-dir results/experiments/comparison
```

LiDAR BEV playback and model rollout GIFs: see `scripts/simulate_lidar_bev.py` and `scripts/simulate_model_rollout_bev.py`.

### HPC (Slurm)

- Stage 1 (LSTM train + val/test eval): `./slurm/submit_stage1_pipeline.sh` — see `slurm/README.md`
- Full three-model pipeline (train + test eval + comparison): `sbatch slurm/run_friday_pipeline.sbatch` or `bash scripts/run_friday_pipeline.sh`

Set `PROJECT_DIR` to your clone path if it is not the default in the sbatch files.

---

## Reproducibility

What you can reproduce from this repo alone:

| Artifact | How |
|----------|-----|
| Train/val/test splits | `data/infos_*.pkl` + `data/manifest.json`, or `scripts/build_expanded_data_splits.py` |
| Stage 1 LSTM | `train/train_lstm_baseline.py` or `slurm/submit_stage1_pipeline.sh` |
| LSTM + Transformer + multi-agent LSTM | `scripts/run_friday_pipeline.sh` (or Slurm wrapper) |
| Test ADE/FDE + qualitative PNGs | `traj_code/evaluation/evaluate_trajectory_checkpoint.py` |
| Model comparison figures | `scripts/plot_model_comparison.py` |
| EDA figures | `eda/EDA.py`, `eda/EDA_pointcloud.py` |
| RF / XGBoost baselines | `train/train_traditional_models.py` |

**You must retrain to obtain checkpoints** — `results/` is not versioned. For paper-grade numbers, use full window counts (avoid tiny `--max-*-windows` debug settings) and fix `--seed` (default `42` in trainers).

**Not wired into the default Friday pipeline** (but implemented in code): `multi_lstm_attn`, `lstm_multimodal`. Train them explicitly with `train/train_trajectory_model.py --model-type ...`.

**External requirements:** processed Waymo data on shared storage; GPU recommended for deep models; LiDAR scripts need `.npy` frames at paths referenced in the infos pickles.

---

## Outputs

Typical paths (all under gitignored `results/`):

- `results/experiments/<run_tag>/{lstm,transformer,multi_lstm}/` — checkpoints, `train_history.csv`, `test_metrics.json`, `qualitative_test/`
- `results/experiments/<run_tag>/comparison/` — ADE/FDE and validation-loss plots
- `results/eda_*`, `results/pointcloud_eda_*` — EDA exports
- `results/lidar_sim/`, `results/model_rollout_sim/` — animations
