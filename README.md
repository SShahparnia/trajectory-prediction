# Trajectory Prediction (Waymo)

**Bird's-Eye Trajectory Forecasting on Waymo** — CMPE project, San Jose State University.

**Team:** Shervan Shahparnia, Aaron Sam, Bhavdeep Randhawa, Atilla Sayin (Department of Computer Engineering, SJSU).

We built an end-to-end pipeline on the [Waymo Open Dataset](https://waymo.com/open/) that turns processed sequences into fixed-horizon bird's-eye-view (BEV) trajectory windows, then trains and evaluates several forecasters under the same splits and metrics (ADE/FDE). Deep models include LSTM, Transformer, and Multi-Agent LSTM; traditional baselines are Random Forest and XGBoost on engineered kinematic features. Deep learning consistently beats the classical baselines in our experiments, with Multi-Agent LSTM strongest when neighbor context is used. The work also reflects the practical cost of large driving datasets: heavy preprocessing and long GPU training on HPC.

This repository holds the code, Slurm jobs, and split metadata pickles. Checkpoints and figures live under `results/` (gitignored). Layout: core library `traj_code/`, entry points in `train/`, `scripts/`, and `eda/`.

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

Dependencies are listed in `requirements.txt` (PyTorch, scikit-learn, XGBoost, matplotlib, etc.).

### Data

**Waymo processed data is not included in this repository.** You must download or mount it yourself (course scratch storage, your own disk, etc.). What *is* included are split **infos** pickles under `data/` that reference frames on shared storage.

Training uses processed Waymo **infos** pickles (`*_infos_*.pkl`), not raw TFRecords:

- `data/infos_train.pkl`
- `data/infos_val.pkl`
- `data/infos_test.pkl`

Each entry may list LiDAR `.npy` paths (e.g. under `waymo_processed_data_v0_5_0/` on `/scratch/...`). You need that tree available locally or mounted for point-cloud EDA and BEV simulations. Trajectory training from the shipped pickles only needs the pickles and the box annotations inside them.

To rebuild splits from full HPC train+val infos (requires those files on your machine):

```bash
python scripts/build_expanded_data_splits.py \
  --train-infos /path/to/waymo_processed_data_v0_5_0_infos_train.pkl \
  --val-infos /path/to/waymo_processed_data_v0_5_0_infos_val.pkl
```

---

## Usage

### Exploratory analysis

Object-level stats and plots (uses shipped `data/infos_*.pkl` only):

```bash
python eda/EDA.py \
  --infos data/infos_train.pkl \
  --max-samples 500 \
  --out-dir results/eda_train_500
```

Point-cloud EDA (requires your own processed `.npy` tree; replace `--processed-root` with your path):

```bash
python eda/EDA_pointcloud.py \
  --infos data/infos_train.pkl \
  --processed-root /path/to/waymo_processed_data_v0_5_0 \
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

### Traditional ML baselines (Random Forest and XGBoost)

```bash
python train/train_traditional_models.py
```

Trains Random Forest and XGBoost on 66 engineered kinematic features from `data/infos_*.pkl` (same windowing as the deep models). Writes under `results/train_random_forest/` and `results/train_xgboost/`.

### Comparison plots and simulations

```bash
python scripts/plot_model_comparison.py \
  --run lstm results/experiments/lstm/train_history.csv results/experiments/lstm/test_metrics.json \
  --run transformer results/experiments/transformer/train_history.csv results/experiments/transformer/test_metrics.json \
  --out-dir results/experiments/comparison
```

LiDAR BEV playback and model rollout GIFs (require processed LiDAR on disk): `scripts/simulate_lidar_bev.py` and `scripts/simulate_model_rollout_bev.py`.

### HPC (Slurm)

- Stage 1 (LSTM train + val/test eval): `./slurm/submit_stage1_pipeline.sh` — see `slurm/README.md`
- Full three-model pipeline (train + test eval + comparison): `sbatch slurm/run_friday_pipeline.sbatch` or `bash scripts/run_friday_pipeline.sh`

Slurm jobs default to course scratch paths for infos and `PROJECT_DIR`; override those in `slurm/pipeline_defaults.env` or at submit time. Set `PROJECT_DIR` in `scripts/run_friday_pipeline.sh` if your clone path differs.

---

## Reproducibility

| Artifact | How |
|----------|-----|
| Train/val/test splits | `data/infos_*.pkl` + `data/manifest.json`, or `scripts/build_expanded_data_splits.py` |
| Stage 1 LSTM | `train/train_lstm_baseline.py` or `slurm/submit_stage1_pipeline.sh` |
| LSTM + Transformer + multi-agent LSTM | `scripts/run_friday_pipeline.sh` (or Slurm wrapper) |
| Test ADE/FDE + qualitative PNGs | `traj_code/evaluation/evaluate_trajectory_checkpoint.py` |
| Model comparison figures | `scripts/plot_model_comparison.py` |
| EDA figures | `eda/EDA.py`, `eda/EDA_pointcloud.py` |
| RF / XGBoost baselines | `train/train_traditional_models.py` |

### `results/` and external data

**`results/` is not in git.** A fresh clone has no checkpoints, metrics, EDA plots, or animations. To see them, either run the commands above or copy an existing `results/` tree from HPC or another machine.

**Additional Waymo assets are your responsibility** (see **Data**): full processed LiDAR (`.npy` under `waymo_processed_data_v0_5_0/`), and optionally the larger HPC train/val infos for split rebuilds. Point-cloud EDA, BEV simulations, and LiDAR-based scripts need that data; training from the shipped `data/infos_*.pkl` does not. A GPU is recommended for deep-model training.

---

## AI usage disclaimer

We used [Cursor](https://cursor.com) only as an editor assistant for **repository housekeeping**: reorganizing folders and file layout, and editing this README for clarity and structure. Cursor was **not** used to design experiments, write model or training logic, generate research results, or author project reports. All trajectory models, data pipelines, evaluation code, and scientific conclusions are the work of the project team.

---

## Outputs

Typical paths (all under gitignored `results/`):

- `results/experiments/<run_tag>/{lstm,transformer,multi_lstm}/` — checkpoints, `train_history.csv`, `test_metrics.json`, `qualitative_test/`
- `results/experiments/<run_tag>/comparison/` — ADE/FDE and validation-loss plots
- `results/eda_*`, `results/pointcloud_eda_*` — EDA exports
- `results/lidar_sim/`, `results/model_rollout_sim/` — animations
