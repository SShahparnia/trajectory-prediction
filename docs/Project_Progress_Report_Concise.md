# Project Progress Report: Trajectory Prediction for Autonomous Driving

**Date:** March 27, 2026 | **Deadline:** March 28, 2026, 4:59 PM  

**Team:** Shervan Shahparnia, Aaron Sam, Bhavdeep Randhawa, Atilla Sayin  

---

## 1. Project Overview & Objectives

We are building ML models for **motion forecasting**: predict an agent’s **future trajectory** (2D positions) from **past motion**, to support safer planning and lower collision risk in dense traffic.

**Evaluation:** **ADE / FDE** (Average and Final Displacement Error) **per agent type** where labels allow, plus **qualitative** trajectory–map or BEV visualizations.

**Plan:**

1. **Baseline:** Single-agent model (**LSTM** or **Transformer**) on past motion only.  
2. **Multi-agent:** Add nearby agents’ **kinematics** (e.g. positions, velocities) and **interaction-aware** structure (padding/masking for variable counts).  
3. **Map-aware (last):** Add **lane / boundary** (or drivable-space) features so predictions align with road structure—**planned as the final stage** if we can obtain **applicable supplementary data** (e.g. Motion map features, vector lanes from TFRecords, or a LiDAR-derived geometric prior); otherwise we prioritize baseline and multi-agent on the processed data we have.

*(Proposal target: [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/); on HPC we currently use course **processed** infos + LiDAR paths from the Waymo mirror, with **Motion TFRecord parsing** or formal scope to processed tracks still to finalize.)*

---

## 2. Dataset & Preprocessing Progress

**Source:** **Waymo Open Dataset** on HPC at `/scratch/lts-data/cmpe249-fa22/Waymo132`: **TFRecords**, preprocessed **LiDAR** (`.npy` under `waymo_processed_data_v0_5_0/`), and **metadata** (`.pkl` infos). We **mapped the folder tree** (TFRecords, processed segments, `trainall/`, `ImageSets/`, train/val info pickles); on **NFS**, full-tree **`du` could hang**, so we used **`find`-based counts** to verify scale (~1k TFRecords, many `.npy` files).

**EDA findings:** **4,761** processed frames in total (**3,769** in the original train infos + **992** in the original val infos)—not “train only.” Roughly **~200k** LiDAR points per frame and **~25** objects per frame (sampled aggregates), **strong class imbalance** (vehicles dominate), and **high variability** in point density and intensity.

**Repository layout:** `data/raw/waymo132` **symlinks** to scratch (no duplicate raw bytes). We **merged** train+val infos and **re-split** by `lidar_sequence` **70% / 15% / 15%** train/val/test → `data/infos_{train,val,test}.pkl` + `manifest.json`. **Expanded EDA** on that split is **in progress** (v1 EDA used train-only infos).

**Completed preprocessing:** Python env / `requirements.txt`, **HPC validation**, **trajectory windows** + **local** coordinate normalization (`waymo_windows.py`), **80/20** train/val in the baseline training script, **feature-engineering** stub, point-cloud subsampling for plots, split/inventory scripts (`count_dataset_entries.py`, `build_expanded_data_splits.py`, `data/README.txt`).

**Pending tasks:** Official **Motion** TFRecord parsing (or explicit scope to **processed** xy), **full benchmark** alignment with Motion splits, **training-time** imbalance handling beyond per-class metrics, and **map** feature integration **only if** suitable supplementary map/lane data becomes available (see §1 stage 3).

---

## 3. Current Status & Challenges

| Component | Status |
|-----------|--------|
| Repo & EDA (v1) | **Done** |
| HPC path, symlink, merged 4,761-frame splits | **Done** |
| Expanded EDA on merged splits | **In progress** |
| Data pipeline (windows) | **Prototype done** |
| Baseline model | **Implemented** |
| Training script | **Implemented** |
| Eval metrics (ADE/FDE) | **Implemented** |
| Multi-agent | **Not started** |
| Map-aware (stage 3) | **Not started** — **conditional** on supplementary map/lane data |

**Active challenges:** **Dataset mismatch** (Motion vs processed HPC data), **HPC limits** (Python 3.6, `waymo_open_dataset` install issues), **filesystem complexity** (large NFS tree; `find` vs `du`), **coordinate alignment** (local vs map/ego frames), and **long-horizon** forecasting difficulty with **variable agent counts** and **class imbalance**.

---

## 4. Team Roles & Roadmap

| Member | Focus area & contributions |
|--------|------------------------------|
| **Aaron** | Data pipeline, HPC paths, EDA, documentation, splits |
| **Bhavdeep** | Feature engineering; map-related features (with Aaron / Shervan) |
| **Shervan** | Models (LSTM baseline), training pipeline, later multi-agent / map-aware |
| **Attila** | Evaluation metrics (ADE/FDE), visualizations, result interpretation |

**Next steps:** Finalize the **dataset pipeline** (Motion vs processed; loaders vs `data/infos_*.pkl`), run **baseline experiments** on HPC, and **evaluate** (ADE/FDE, per-class where possible).

**Timeline:** **Baseline complete** (early April) → **Multi-agent model** (mid April) → **Map-aware model** (late April, **if applicable supplementary data is available**) → **Final report / presentation** (final weeks).

---

## 5. Appendix

**Figures:** `class_counts_top10.png`, `objects_per_frame_hist.png`, `speed_norm_hist.png`, `num_points_per_frame_hist.png`, `mean_intensity_per_frame_hist.png`, `bev_00000.png`, `lidar3d_00000.png`, `pointcloud_timeline_3d.png` (under `results/eda_full/`, `results/views_top_bottom/`, etc., as applicable).

**References:** [Waymo Open Dataset](https://waymo.com/open/)
