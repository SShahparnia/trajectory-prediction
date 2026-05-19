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
- `data_pipeline/multi_agent_windows.py`: builds multi-agent windows (ego + K neighbors + mask)
- `feature_engineering/trajectory_features.py`: simple feature helper (`xy -> [xy, velocity]`)
- `models/lstm_baseline.py`: baseline LSTM encoder + MLP prediction head
- `models/transformer_baseline.py`: transformer encoder baseline for single-agent forecasting
- `models/multi_agent_lstm.py`: interaction-aware multi-agent LSTM with masked neighbor pooling
- `evaluation/metrics.py`: ADE / FDE
- `evaluation/evaluate_checkpoint.py`: evaluate a saved checkpoint
- `evaluation/evaluate_trajectory_checkpoint.py`: unified checkpoint evaluator + qualitative trajectory plots

Top-level scripts:
- `EDA.py`: object-level EDA from `*_infos_*.pkl`
- `EDA_pointcloud.py`: point-cloud EDA from processed `.npy` frames
- `train/train_lstm_baseline.py`: trains baseline and saves `best_model.pt`
- `train/train_trajectory_model.py`: unified trainer for `lstm`, `transformer`, and `multi_lstm`
- `scripts/plot_model_comparison.py`: creates ADE/FDE + validation-loss comparison figures
- `scripts/simulate_lidar_bev.py`: LiDAR BEV/3D playback with optional GT/cluster overlays
- `scripts/simulate_model_rollout_bev.py`: model-driven rollout animation (past/GT/pred, optional neighbors)
- `scripts/run_friday_pipeline.sh`: one-command local full pipeline runner

Output folders:
- `results/eda_*`
- `results/pointcloud_eda*`
- `results/experiments/<run_tag>/` (canonical train/val/test pipeline output)
- `results/lidar_sim/*` (LiDAR scene playback assets)
- `results/model_rollout_sim/*` (model rollout simulation GIFs)

Slurm:
- `slurm/submit_stage1_pipeline.sh`: submit train + val/test eval pipeline
- `slurm/train_stage1_lstm.sbatch`: Stage 1 train job
- `slurm/eval_checkpoint.sbatch`: checkpoint evaluation job
- `slurm/pipeline_defaults.env`: shared defaults
- `slurm/run_friday_pipeline.sbatch`: CPU-friendly one-job full pipeline submit

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

1. Develop/debug with small settings (`max-windows`, `max-samples` low).
2. Submit full Stage 1 pipeline (train + val + test) via Slurm.
3. Keep each run in `results/experiments/<run_tag>/`.

Preferred path:

```bash
./slurm/submit_stage1_pipeline.sh
```

Details and overrides are documented in `slurm/README.md`.

---

## Next Milestones (Team Plan)

- Baseline complete: single-agent LSTM on XY trajectories (current)
- Multi-agent context: add nearby agent features + masking
- Map-aware model: lane and boundary features
- Better eval suite: per-class ADE/FDE, trajectory visual overlays, failure case reports

---

## Fast Friday Milestone (LSTM + Transformer + Multi-Agent)

If you only need the due-soon scope, use your existing split pickles in `data/`:

- `data/infos_train.pkl`
- `data/infos_val.pkl`
- `data/infos_test.pkl`

Train 3 model variants on the same split protocol:

```bash
source .venv/bin/activate

# 1) Single-agent LSTM
python train/train_trajectory_model.py \
  --model-type lstm \
  --train-infos data/infos_train.pkl \
  --val-infos data/infos_val.pkl \
  --past-len 10 --future-len 20 \
  --max-train-windows 2500 --max-val-windows 800 \
  --epochs 12 --batch-size 64 \
  --out-dir results/experiments/lstm

# 2) Single-agent Transformer
python train/train_trajectory_model.py \
  --model-type transformer \
  --train-infos data/infos_train.pkl \
  --val-infos data/infos_val.pkl \
  --past-len 10 --future-len 20 \
  --max-train-windows 2500 --max-val-windows 800 \
  --epochs 12 --batch-size 64 \
  --out-dir results/experiments/transformer

# 3) Multi-agent LSTM (neighbor masking/padding)
python train/train_trajectory_model.py \
  --model-type multi_lstm \
  --train-infos data/infos_train.pkl \
  --val-infos data/infos_val.pkl \
  --past-len 10 --future-len 20 \
  --max-neighbors 12 \
  --max-train-windows 2500 --max-val-windows 800 \
  --epochs 12 --batch-size 48 \
  --out-dir results/experiments/multi_lstm
```

Evaluate on test split:

```bash
python code/evaluation/evaluate_trajectory_checkpoint.py \
  --checkpoint results/experiments/lstm/best_model.pt \
  --infos data/infos_test.pkl \
  --past-len 10 --future-len 20 \
  --max-windows 800 \
  --metrics-out results/experiments/lstm/test_metrics.json \
  --viz-out-dir results/experiments/lstm/qualitative_test \
  --num-viz 12

python code/evaluation/evaluate_trajectory_checkpoint.py \
  --checkpoint results/experiments/transformer/best_model.pt \
  --infos data/infos_test.pkl \
  --past-len 10 --future-len 20 \
  --max-windows 800 \
  --metrics-out results/experiments/transformer/test_metrics.json \
  --viz-out-dir results/experiments/transformer/qualitative_test \
  --num-viz 12

python code/evaluation/evaluate_trajectory_checkpoint.py \
  --checkpoint results/experiments/multi_lstm/best_model.pt \
  --infos data/infos_test.pkl \
  --past-len 10 --future-len 20 \
  --max-neighbors 12 \
  --max-windows 800 \
  --metrics-out results/experiments/multi_lstm/test_metrics.json \
  --viz-out-dir results/experiments/multi_lstm/qualitative_test \
  --num-viz 12
```

Create side-by-side comparison plots:

```bash
python scripts/plot_model_comparison.py \
  --run lstm results/experiments/lstm/train_history.csv results/experiments/lstm/test_metrics.json \
  --run transformer results/experiments/transformer/train_history.csv results/experiments/transformer/test_metrics.json \
  --run multi_lstm results/experiments/multi_lstm/train_history.csv results/experiments/multi_lstm/test_metrics.json \
  --out-dir results/experiments/comparison
```

Outputs:
- `results/experiments/comparison/val_loss_comparison.png`
- `results/experiments/comparison/ade_fde_comparison.png`
- `results/experiments/comparison/metrics_summary.csv`

### One-command run (local shell)

```bash
chmod +x scripts/run_friday_pipeline.sh
bash scripts/run_friday_pipeline.sh
```

### One-command run (Slurm)

```bash
sbatch slurm/run_friday_pipeline.sbatch
```

Optional overrides at submit time:

```bash
RUN_TAG=friday_final \
MAX_TRAIN_WINDOWS=6000 \
MAX_VAL_WINDOWS=2000 \
MAX_TEST_WINDOWS=2000 \
EPOCHS=12 \
sbatch slurm/run_friday_pipeline.sbatch
```

Additional qualitative visuals are saved in:
- `results/experiments/<run_tag>/lstm/qualitative_test/`
- `results/experiments/<run_tag>/transformer/qualitative_test/`
- `results/experiments/<run_tag>/multi_lstm/qualitative_test/`

### LiDAR simulation-style playback (BEV or 3D)

Generate a frame-by-frame BEV animation from processed LiDAR:

```bash
source .venv/bin/activate
python scripts/simulate_lidar_bev.py \
  --infos data/infos_test.pkl \
  --processed-root /scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0 \
  --num-frames 30 \
  --fps 5 \
  --overlay-gt \
  --cluster-detections \
  --out-dir results/lidar_sim/demo1
```

Clean top-down presentation version:

```bash
python scripts/simulate_lidar_bev.py \
  --infos data/infos_test.pkl \
  --processed-root /scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0 \
  --num-frames 30 \
  --fps 5 \
  --clean \
  --out-dir results/lidar_sim/demo1_clean
```

3D clean playback:

```bash
python scripts/simulate_lidar_bev.py \
  --infos data/infos_test.pkl \
  --processed-root /scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0 \
  --mode 3d \
  --num-frames 30 \
  --fps 5 \
  --clean \
  --out-dir results/lidar_sim/demo1_3d_clean
```

Outputs:
- `results/lidar_sim/demo1/frames/*.png`
- `results/lidar_sim/demo1/lidar_bev_sim.gif`

Notes:
- `--overlay-gt` shows GT object centers from annotations.
- `--cluster-detections` shows unsupervised DBSCAN cluster centroids as a lightweight object-detection proxy.

### Model rollout simulation (prediction GIFs)

Generate a rollout animation from a trained checkpoint:

```bash
python scripts/simulate_model_rollout_bev.py \
  --checkpoint results/experiments/friday_20260429_230906/multi_lstm/best_model.pt \
  --infos data/infos_test.pkl \
  --past-len 10 \
  --future-len 20 \
  --max-windows 800 \
  --max-neighbors 12 \
  --sample-idx -1 \
  --fps 5 \
  --out-dir results/model_rollout_sim/demo_multi
```

Comparable clean overhead rollouts (same sample across models):

```bash
python scripts/simulate_model_rollout_bev.py \
  --checkpoint results/experiments/friday_20260429_230906/lstm/best_model.pt \
  --infos data/infos_test.pkl \
  --past-len 10 --future-len 20 \
  --max-windows 220 \
  --sample-idx 163 \
  --clean \
  --out-dir results/model_rollout_sim/overhead_lstm
```

Legend meaning:
- `past` (black): model input history
- `future_gt` (green): ground-truth future
- `future_pred` (red dashed): model prediction

---

## Common Pitfalls

- If `pip install -r requirements.txt` fails, confirm venv is activated.
- If training builds no windows, reduce `--past-len`/`--future-len` or increase `--max-windows` search scope.
- If paths fail on compute nodes, verify shared filesystem mount for `/scratch`.
- If GPU is unavailable, script falls back to CPU automatically (slower).
