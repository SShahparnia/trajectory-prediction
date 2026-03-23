# Trajectory Prediction for Autonomous Driving

End-to-end project scaffold for trajectory prediction on Waymo data:
- Data pipeline (window builder from Waymo processed infos)
- Feature engineering utilities
- Baseline LSTM model
- Training + checkpoint save
- Evaluation (ADE/FDE)
- EDA scripts for metadata and point clouds

This repository is set up so your team can train **your own model and weights** on HPC data, not rely on someone else's trained outputs.

---

## Project Layout

`code/`
- `data_pipeline/waymo_windows.py`: builds supervised trajectory windows from Waymo processed `*_infos_*.pkl`
- `feature_engineering/trajectory_features.py`: simple feature helper (`xy -> [xy, velocity]`)
- `models/lstm_baseline.py`: baseline LSTM encoder + MLP prediction head
- `evaluation/metrics.py`: ADE / FDE
- `evaluation/evaluate_checkpoint.py`: evaluate a saved checkpoint

Top-level scripts:
- `EDA.py`: object-level EDA from `*_infos_*.pkl`
- `EDA_pointcloud.py`: point-cloud EDA from processed `.npy` frames
- `train/train_lstm_baseline.py`: trains baseline and saves `best_model.pt`

Output folders:
- `results/eda_*`
- `results/pointcloud_eda*`
- `results/train_lstm_baseline*`

---

## HPC Data Notes (Brief)

Your Waymo data on HPC is under:

`/scratch/lts-data/cmpe249-fa22/Waymo132`

Key subfolders/files:
- Raw-style shards: `training_0000 ... training_0031`, `validation_0000 ... validation_0007`
- Processed data root: `waymo_processed_data_v0_5_0/`
- Processed metadata:
  - `waymo_processed_data_v0_5_0_infos_train.pkl`
  - `waymo_processed_data_v0_5_0_infos_val.pkl`

Important clarification:
- `*_infos_*.pkl` are **metadata/preprocessing artifacts**, not trained model weights.
- Your model weights are created by your training script and saved as `.pt` (PyTorch checkpoint).

---

## Environment Setup

### 1) Create and activate venv

```bash
cd /home/017264195/trajectory-prediction
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Current required packages (in `requirements.txt`):
- numpy
- pandas
- torch
- tensorflow
- scikit-learn
- matplotlib
- jupyter
- pytest
- tqdm
- h5py

Note:
- `pickle`, `os`, `sys`, `argparse`, etc. are Python standard library modules and are not added to `requirements.txt`.

---

## Quick Data Sanity Checks

From login node:

```bash
ls -lah /scratch/lts-data/cmpe249-fa22/Waymo132
du -sh /scratch/lts-data/cmpe249-fa22/Waymo132
```

Check processed train infos exists:

```bash
ls -lh /scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0_infos_train.pkl
```

---

## EDA: Metadata / Labels

Run object-level EDA:

```bash
source .venv/bin/activate
python EDA.py \
  --infos /scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0_infos_train.pkl \
  --max-samples 500 \
  --out-dir results/eda_train_500
```

Outputs:
- `frames_summary.csv`
- `class_counts.csv`
- `class_counts_top10.png`
- `objects_per_frame_hist.png`
- `num_points_per_object_hist.png`
- `speed_norm_hist.png`

---

## EDA: Point Clouds

Run point-cloud EDA:

```bash
source .venv/bin/activate
python EDA_pointcloud.py \
  --infos /scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0_infos_train.pkl \
  --processed-root /scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0 \
  --max-samples 200 \
  --plot-samples 5 \
  --plot-3d-samples 3 \
  --3d-point-cap 100000 \
  --timeline-frames 16 \
  --out-dir results/pointcloud_eda_train_200
```

Outputs include:
- per-frame point stats CSV
- BEV samples
- combined timeline image: `pointcloud_timeline_3d.png`
- optional **single-scan 3D** images in `lidar_3d_samples/` when you pass `--plot-3d-samples N` (one LiDAR frame as x–y–z scatter, colored by height)
- **`bev_samples/`** = **true top-down map**: 2D scatter of **x vs y** (height **z** only affects color)
- add **`--extra-ortho-views`** with `--plot-3d-samples` to also get **`lidar3d_*_view_top.png`** (camera above, looking down) and **`lidar3d_*_view_bottom.png`** (camera below, looking up)

**Longer 3D timelines (more than ~16 frames):**

- Set `--timeline-frames` to any count you want (e.g. `60`, `120`).
- Raise `--timeline-point-cap` if the plot looks too sparse (more frames × fewer points per color).
- By default the timeline only sees the first `--max-samples` infos rows, so a long run may need either a larger `--max-samples` or **`--timeline-scan-full-infos`** so the timeline uses the full pickle while CSV/BEV still use a modest `--max-samples`.
- Use **`--timeline-frames 0`** (or `-1`) to include **all** frames of the chosen sequence present in that infos list.
- Pin the segment with **`--timeline-sequence segment-..._with_camera_labels`**.

---

## Train Your Baseline Model (Create Your Own Weights)

This script builds trajectory windows from your Waymo processed infos and trains LSTM.

```bash
source .venv/bin/activate
python train/train_lstm_baseline.py \
  --infos /scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0_infos_train.pkl \
  --past-len 10 \
  --future-len 20 \
  --max-windows 12000 \
  --epochs 6 \
  --batch-size 128 \
  --out-dir results/train_lstm_baseline
```

Training outputs:
- `results/train_lstm_baseline/best_model.pt`
- `results/train_lstm_baseline/train_history.csv`

These are **your** model artifacts.

---

## Evaluate a Saved Checkpoint

```bash
source .venv/bin/activate
python code/evaluation/evaluate_checkpoint.py \
  --checkpoint results/train_lstm_baseline/best_model.pt \
  --infos /scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0_infos_train.pkl \
  --past-len 10 \
  --future-len 20 \
  --max-windows 6000
```

Prints:
- `ADE`
- `FDE`

---

## Suggested HPC Workflow

1. Develop/debug on login node with small settings (`max-windows`, `max-samples` low).
2. Run full training via scheduled job for heavier runs.
3. Save outputs under `results/` with experiment-specific folder names.

Example batch script (adjust partition/account to your cluster):

```bash
#!/bin/bash
#SBATCH --job-name=traj-lstm
#SBATCH --output=logs/train_lstm_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

cd /home/017264195/trajectory-prediction
source .venv/bin/activate

python train/train_lstm_baseline.py \
  --infos /scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0_infos_train.pkl \
  --past-len 10 \
  --future-len 20 \
  --max-windows 50000 \
  --epochs 20 \
  --batch-size 256 \
  --out-dir results/train_lstm_baseline_full
```

Submit with:

```bash
sbatch <your_script>.sh
```

---

## Next Milestones (Team Plan)

- Baseline complete: single-agent LSTM on XY trajectories (current)
- Multi-agent context: add nearby agent features + masking
- Map-aware model: lane and boundary features
- Better eval suite: per-class ADE/FDE, trajectory visual overlays, failure case reports

---

## Common Pitfalls

- If `pip install -r requirements.txt` fails, confirm venv is activated.
- If training builds no windows, reduce `--past-len`/`--future-len` or increase `--max-windows` search scope.
- If paths fail on compute nodes, verify shared filesystem mount for `/scratch`.
- If GPU is unavailable, script falls back to CPU automatically (slower).
