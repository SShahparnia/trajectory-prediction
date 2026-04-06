# Project Progress Report

**Course project - Trajectory Prediction for Autonomous Driving**

**Team:** Aaron Sam, Shervan Shahparnia, Bhavdeep Randhawa, Attila Sayin  

**Report date:** March 27, 2026  

**Submission deadline (assignment):** Saturday, March 28, 2026, 4:59 PM  

---

## 1. Project Overview

### Topic and goal

We are building **machine learning models for motion forecasting**: given a short **history of past positions** (and eventually richer context), predict each agent's **future trajectory** over the next several seconds. The problem is central to autonomous driving: accurate forecasts reduce collision risk and support planning in dense traffic.

### Problem statement

**Input:** Recent motion of a target agent (baseline stage: 2D positions over a fixed past horizon).  

**Output:** Predicted future positions over a future horizon (same dimensionality as training targets).  

**Why it matters:** Poor trajectory prediction can lead to unsafe maneuvers; realistic forecasts that respect road structure and multi-agent interaction are needed for self-driving stacks, simulation, and safety analysis.

### Planned evolution (three stages)

1. **Baseline:** Single-agent model (LSTM or Transformer encoder) on past motion only.  
2. **Multi-agent:** Incorporate nearby agents' positions/velocities and interaction-aware structure.  
3. **Map-aware (last):** Add lane/boundary and map features so predictions align with drivable space. We treat this as the **final stage**, contingent on **applicable supplementary data** (e.g. Motion map features from TFRecords, or a LiDAR-derived geometric prior if vector maps remain unavailable).

**Evaluation:** We plan to use **Average Displacement Error (ADE)** and **Final Displacement Error (FDE)** per agent type (vehicle, pedestrian, cyclist), plus qualitative trajectory-map visualizations.

---

## 2. Dataset & Preprocessing Progress

**Source:** **Waymo Open Dataset** on HPC at `/scratch/lts-data/cmpe249-fa22/Waymo132`: **TFRecords** (training/validation shards), **preprocessed LiDAR** as **`.npy`** under `waymo_processed_data_v0_5_0/`, and **metadata** in **`.pkl`** info lists. Locating the canonical mirror required **tracing nested folder trees** (TFRecords, processed segments, `trainall/`, `ImageSets/`, original `*_infos_train.pkl` / `*_infos_val.pkl`). On **NFS**, **`du` over the full tree could hang**, so we used **`find`-based inventories** (and optional targeted `du`) to confirm scale—**~1k** TFRecords, **tens of thousands** of `.npy` files, and hundreds of ancillary pickles—not a stub copy.

**EDA findings (beyond “3,769 training rows”):** The **original course train pickle** has **3,769** frames; the **original val pickle** adds **992** more. Together they define **4,761** processed frames with LiDAR pointers and annotations. Aggregated EDA (train-focused v1; expanded pass ongoing) shows on the order of **~200,000** LiDAR points per frame (typical), **~25** objects per frame (sampled), **strong class imbalance** (vehicles—and often signage in 3D labels—dominate vulnerable road users), and **high variability** in point counts and intensity across frames (density and return strength are not uniform).

**Repository vs scratch:** **`data/raw/waymo132`** is a **symlink** to the scratch tree—**no duplicate** multi-terabyte raw data in the repo. We **merged** train+val infos and **re-split** all **4,761** frames **by `lidar_sequence`** into **70% / 15% / 15%** train/validation/test (`data/infos_train.pkl`, `infos_val.pkl`, `infos_test.pkl`, **`manifest.json`**; `scripts/build_expanded_data_splits.py`). We are **extending EDA** to this **expanded** split so tables and plots match the frame universe we train on.

**Completed preprocessing:** Python virtual environment and **`requirements.txt`**, **HPC access validation** and documented paths, **trajectory window construction** (sequence-aligned past/future pairs, **local** coordinate normalization in `waymo_windows.py`), **80/20** random train/val inside `train_lstm_baseline.py`, **feature-engineering** scaffolding (`trajectory_features.py`), point-cloud **subsampling** for visualization, and **split/inventory** tooling (`count_dataset_entries.py`, `data/README.txt`).

**Pending tasks:** Official **Waymo Motion** TFRecord decoding (or a documented decision to stay on processed tracking-style xy), **full benchmark alignment** with motion splits where applicable, **training-time** imbalance handling beyond reporting per-class metrics, and **map** feature integration **as the last step** when/if we can source **supplementary** lane or map-aligned data (see stage 3 above).

---

## 3. Dataset Investigation and Analysis So Far

### Dataset source

We use data from the **Waymo Open Dataset** ecosystem available on our **HPC shared filesystem**. The on-cluster location we verified is:

`/scratch/lts-data/cmpe249-fa22/Waymo132`

This tree includes:

- **Perception-style TFRecord shards** (e.g. `training_*`, `validation_*`, `segment-*_with_camera_labels.tfrecord`).
- **Preprocessed** assets under `waymo_processed_data_v0_5_0/`, including per-frame **LiDAR** stored as `.npy` files and **metadata** in Python pickle files such as `waymo_processed_data_v0_5_0_infos_train.pkl` and `waymo_processed_data_v0_5_0_infos_val.pkl`.

Our **course proposal** references the **Waymo Open Motion Dataset** (motion forecasting). The work completed so far primarily **engages processed Waymo perception-style infos and point clouds** on HPC to validate access, scale, and label structure; **full Motion TFRecord ingestion** is identified as the next step for strict alignment with the motion benchmark.

### Tracing the HPC folder tree and data-layout discoveries

The shared course mirror is **not** a single flat folder. We had to **walk the tree systematically** to connect pieces: TFRecord shards (`training_*`, `validation_*`), the **`waymo_processed_data_v0_5_0/`** subtree (per-segment folders of **`.npy`** LiDAR and paths referenced from metadata), **`trainall/`**, **`ImageSets/`**, and the **pickled info lists** (`waymo_processed_data_v0_5_0_infos_train.pkl`, `..._val.pkl`). Early confusion came from **directory listing “sizes”** vs **actual disk usage**; on **NFS**, **`du` on the full tree could hang**, so we used **`find`-based counts** (and optional `du` with care) to inventory **~1k** TFRecord files, **tens of thousands** of `.npy` scans, and **hundreds** of ancillary pickles—confirming we were pointed at the full mirror, not a stub.

**Key discovery:** The **original train and validation info pickles together** list **4,761** frames (3,769 + 992). That is the **full processed frame set** we care about for modeling, not the train file alone.

**Repository layout (`data/`) vs scratch:** We do **not** duplicate terabytes into the project directory. **`data/raw/waymo132`** is a **symlink** to `/scratch/lts-data/cmpe249-fa22/Waymo132`, so paths under `data/raw/` resolve to the **same bytes** on the HPC filesystem. Separately, we wrote **`data/infos_train.pkl`**, **`infos_val.pkl`**, **`infos_test.pkl`** by **merging** the original train and val infos and **re-splitting** all 4,761 frames **by `lidar_sequence`** into **70% / 15% / 15%** train/validation/test (with a saved **`manifest.json`** for reproducibility). Those pickles are **metadata only**—they list frames and paths; **LiDAR and TFRecords remain on scratch.**

**Scripts supporting this:** `scripts/count_dataset_entries.py` (full-tree file counts; `du` optional) and `scripts/build_expanded_data_splits.py` (build merged splits). **`data/README.txt`** documents raw vs processed vs our splits.

### Scale and samples

From analysis of `waymo_processed_data_v0_5_0_infos_train.pkl` (original course split):

- **Frames / samples (metadata rows):** **3,769** training info entries (each corresponds to a frame with annotations and a pointer to a point cloud file).
- **Merged train+val (expanded corpus):** **4,761** frames total; our **re-split** assigns these into train/val/test as above for experiments that use the **full** processed set.

- **Point clouds:** On the order of **~200,000** points per frame (typical), summarized in our EDA outputs.

### Features and targets (current understanding)

**Per-frame metadata (`infos` pickle), non-exhaustive:**

- **Pose:** 4x4 vehicle pose matrix.  
- **Annotations (`annos`):** object class names, 3D boxes in LiDAR frame (`gt_boxes_lidar`), object IDs, difficulty, global speed/acceleration, point counts in box, etc.  
- **Point cloud pointer:** `lidar_sequence` + `sample_idx` -> path to `.../segment.../####.npy`.

**For our initial trajectory windows** (implemented in code), we derive **supervised pairs** from **per-frame object centers** (`location` xy) tracked across time within a segment: **past window** -> **future window**, expressed in a **local frame** centered on the last past position.

**Target variable:** Future **(x, y)** positions (or displacements in local coordinates) over `future_len` steps.

**Data types:** Mixed - float arrays for geometry and motion, strings/categories for class names, integers for IDs and indices.

### Class balance and patterns

Object-level EDA on the processed train infos showed **strong imbalance** across coarse categories - e.g. **Vehicle** and **Sign** dominated raw counts relative to **Pedestrian** in the summarized class histogram. That implies models trained naively may favor dominant classes unless we use **per-class metrics**, **sampling**, or **class weights**.

We also observed **high object counts per frame** (on the order of **~25** objects per frame in aggregated statistics over sampled frames), which reinforces the need for **variable-length** handling (padding/masking) in later multi-agent stages.

### Missing values, duplicates, outliers

- **Missing LiDAR files:** Our point-cloud EDA script reports **missing file counts** when paths do not resolve; in successful runs we achieved **zero missing** files for the scanned subset.  
- **Duplicates:** We have not run a dedicated duplicate-frame audit; frame IDs are taken from metadata.  
- **Outliers:** Point-cloud summaries include **min/max xyz** and intensity statistics; extreme ranges can indicate noise or sparse returns - useful for downstream filtering later.

### Visualizations and summary statistics completed

We produced **reproducible scripts and outputs** under `results/`:

| Artifact | Purpose |
|----------|---------|
| `EDA.py` | Class counts, objects per frame, points-in-box, speed-norm histograms, CSV summaries |
| `EDA_pointcloud.py` | Per-frame point counts, xyz/intensity summaries, BEV plots, optional single-scan 3D, top/bottom orthographic views, multi-frame 3D timeline |

These go beyond "download only": they quantify **scale**, **class skew**, **density**, and **geometry** of real HPC data.

**Expanded data and ongoing EDA:** The first EDA pass targeted the **original training infos** (and sampled LiDAR paths) to validate the pipeline. After **mapping the full scratch tree** and building **merged 4,761-frame splits**, we are **running and extending EDA** on this **expanded** corpus—e.g. object-level statistics and point-cloud summaries **across train+val coverage** and, where useful, **per-split** checks—so that reported figures and CSVs reflect the **same** frame universe we intend for training, not only the original 3,769-row train pickle. New plots and tables will be added under `results/` as this round completes.

---

## 4. Preprocessing Completed So Far

We have **not** yet completed a full Motion-Dataset-specific pipeline (e.g. official **tfrecord -> tf.example** decoding with Waymo's motion protobufs on this environment). **Completed preprocessing and preparation** includes:

1. **Environment:** Python virtual environment and pinned third-party stack in `requirements.txt` (NumPy, Pandas, PyTorch, TensorFlow, scikit-learn, Matplotlib, Jupyter, pytest, tqdm, h5py).  
2. **Data localization:** Confirmed read access to shared Waymo data on `/scratch/...`, traced subfolders (TFRecords, `waymo_processed_data_v0_5_0/`, infos pickles), documented paths in `README.md`, and added **`data/raw/waymo132`** (symlink) plus **`data/infos_{train,val,test}.pkl`** from merged re-splits (see §§2–3).  
3. **Trajectory window construction (for baseline experiments):**  
   - Group frames by **sequence** and **time index**.  
   - For each anchor time, extract **past** and **future** xy for objects present across the full window.  
   - **Local coordinate normalization:** subtract last past position so the origin is the "current" agent location.  
   - Implemented in `code/data_pipeline/waymo_windows.py` (`build_xy_windows`, `TrajectoryDataset`).  
4. **Train/validation split (for training script):** **80/20** random split inside `train/train_lstm_baseline.py` (seeded for reproducibility).  
5. **Feature engineering stub:** `code/feature_engineering/trajectory_features.py` (e.g. velocity from consecutive positions) - available for extension, not yet wired as the sole input to the baseline.  
6. **Point-cloud subsampling for visualization:** Random caps per frame and per timeline for **fast, faithful exploratory plots** without loading billions of points.

**Not yet completed (honest scope):** official Motion Dataset TFRecord parsing, map raster/vector ingestion, full train/val/test splits by **official** motion splits, systematic duplicate removal, and imbalance mitigation in training loops.

---

## 5. Current Progress

| Area | Status |
|------|--------|
| Repository structure | **Done:** `data_pipeline/`, `feature_engineering/`, `models/`, `evaluation/`, `train/` |
| Exploratory data analysis | **Done (v1):** metadata EDA (`EDA.py`), LiDAR EDA (`EDA_pointcloud.py`), documented in `README.md`. **In progress:** EDA on **expanded** merged splits (`data/infos_*.pkl`) after HPC tree tracing and full-frame inventory |
| HPC data discovery & splits | **Done:** symlink `data/raw/waymo132` → scratch; merged 4,761-frame re-split; `scripts/count_dataset_entries.py`, `build_expanded_data_splits.py`; `data/README.txt` |
| Data loading for trajectory windows | **Done (prototype):** `waymo_windows.py` builds `(past, future)` tensors from processed infos |
| Baseline model | **Implemented:** `LSTMBaseline` in `code/models/lstm_baseline.py` |
| Training script | **Implemented:** `train/train_lstm_baseline.py` saves `best_model.pt` and `train_history.csv` |
| Evaluation metrics | **Implemented:** ADE/FDE in `code/evaluation/metrics.py`; `evaluate_checkpoint.py` for checkpoints |
| Full training runs on HPC GPU jobs | **Optional / team discretion** - pipeline is ready; extensive runs not required for this progress report |
| Multi-agent stage | **Not started** (planned) |
| Map-aware stage | **Not started** (planned; **last**; **conditional** on supplementary map/lane data) |
| Literature review | **Ongoing** as part of proposal and method selection |

---

## 6. Active Challenges

1. **Dataset mismatch:** The proposal emphasizes **Waymo Motion** (motion forecasting TFRecords), while HPC primarily exposes **perception-style** assets and **preprocessed** infos + LiDAR. We must either **integrate Motion parsing** with a supported stack or **formally scope** the baseline to processed tracking-style xy and document the gap to the motion benchmark.  
2. **HPC limitations:** The cluster runs older **Python 3.6**, which caps library versions; **`waymo_open_dataset`** wheels did not install cleanly for us. We lean on **NumPy/pickle** pipelines and preprocessed shards rather than official TFRecord decoding in-process.  
3. **Filesystem complexity:** The Waymo mirror is **deep and large**; **`du` on NFS can stall**, so discovering layout and counts required **walking subtrees** and **`find`-style** inventories instead of a single quick size command.  
4. **Coordinate alignment:** Forecasting is sensitive to **global vs agent-local** frames; we center trajectories in the window builder, but **map-aligned** coordinates and consistent ego vs object frames remain future work.  
5. **Long-horizon prediction under messy supervision:** **Variable agent counts** per frame, **class imbalance**, and **error growth** over long prediction horizons compound—motivating per-class metrics, padding/masking for multi-agent batches, and possibly shorter horizons or multi-scale loss before map features land.

---

## 7. Plan for Completion

**Next steps:** Finalize the **dataset pipeline** (Motion vs processed scope; loaders aligned with `data/infos_*.pkl` where used), run **baseline experiments** on HPC, and **evaluate** with ADE/FDE (and per-class where labels allow).

**Timeline:** **Baseline complete** (early April) → **Multi-agent model** (mid April) → **Map-aware model** (late April, **if applicable supplementary map data is available**) → **Final report / presentation** (final weeks).

### Remaining tasks (detail)

1. **Finalize data path:** Integrate **Waymo Motion** TFRecords with a supported runtime, or formally scope the project to **processed HPC tracking** data and document the choice; wire training to **merged splits** as needed.  
2. **Baseline experiments:** Run `train_lstm_baseline.py` on HPC (CPU/GPU job), tune `past_len`, `future_len`, `max-windows`, learning rate.  
3. **Evaluation:** Report **ADE/FDE** on validation; add **per-class** breakdown once labels are mapped consistently.  
4. **Multi-agent module:** Nearby agent tensor + attention or interaction layer.  
5. **Map features (conditional, last):** Lane/boundary encoding (raster or vector) **only if** we obtain suitable supplementary data; otherwise document LiDAR-BEV or scope limits.  
6. **Figures for final report:** Prediction vs ground truth overlaid on BEV or map.

### Team responsibilities (aligned with proposal)

| Member | Focus | Next steps |
|--------|--------|------------|
| **Aaron** | Data pipeline | Motion vs processed decision; data loaders; splits; documentation |
| **Bhavdeep** | Features & map | Velocities, local frames, map feature design |
| **Shervan** | Models & training | Baseline runs, multi-agent architecture, training stability |
| **Attila** | Evaluation & visualization | ADE/FDE tables, plots, error analysis |

### Methods to try

- **Baseline:** LSTM (implemented); optional **Transformer** encoder if compute allows.  
- **Later:** Graph or attention over agents; **map CNN** or **polyline** encoders if supplementary map or lane data is available.

### Evaluation plan

- **ADE / FDE** (primary).  
- **Per-class** metrics where labels permit.  
- **Qualitative** trajectory plots vs map / BEV.

---

## 8. Team Contributions (so far)

Contributions are aligned with the division of labor in our project proposal and the artifacts in the repository:

- **Aaron:** Data pipeline direction; HPC data path verification; trajectory window builder (`waymo_windows.py`); EDA and documentation (`README.md`, EDA scripts).  
- **Bhavdeep:** Feature-engineering scaffolding (`trajectory_features.py`); coordination on local frames and future map-related features.  
- **Shervan:** Baseline model (`lstm_baseline.py`); training entry point (`train_lstm_baseline.py`); checkpoint format and hyperparameters.  
- **Attila:** Evaluation metrics (`metrics.py`, `evaluate_checkpoint.py`); visualization-heavy EDA (plots, histograms, point-cloud figures) and interpretation support.

*All members participate in design discussions, debugging, and integration; the table reflects primary ownership.*

---

## 9. Figures for the report (EDA support)

*Instructions for the team: paste or embed each image **immediately above** the paragraph that follows its `[filename]` line. Files below are names we used under `results/` (e.g. `results/eda_full/`, `results/views_top_bottom/`); copy or export to your PDF as needed.*

### 9.1 Object- and label-level EDA (`EDA.py`)

`[class_counts_top10.png]`

This bar chart summarizes how often each coarse **object class** appears in the sampled processed **train** frames. It supports our discussion of **class imbalance**: models can become biased toward frequent categories (e.g. vehicles vs. pedestrians) unless we report **per-class** metrics or adjust sampling/weights.

`[objects_per_frame_hist.png]`

This histogram shows the **distribution of object counts per frame**. It supports the claim that scenes are **multi-object** and dense, which motivates **variable-length** inputs and later **multi-agent** modeling with padding or masking.

`[speed_norm_hist.png]` *(optional if space is tight)*

This histogram summarizes **global speed magnitude** of annotated objects (from metadata). It helps characterize **typical motion speeds** in the subset we analyzed and can be cited when discussing motion scale for forecasting.

### 9.2 Point-cloud EDA (`EDA_pointcloud.py`)

`[num_points_per_frame_hist.png]`

This histogram reports **how many LiDAR points** are returned per frame (after loading `.npy` clouds). It documents **sensor density** and scale (~order of magnitude points per scan) for readers who want to understand computational load and sparsity.

`[mean_intensity_per_frame_hist.png]` *(optional)*

This summarizes **mean intensity** per frame across the subset. It is secondary to xyz geometry but shows we inspected **return strength**, not only coordinates.

`[bev_00000.png]` *(or another `bev_*.png` from `bev_samples/`)*

**Bird's-eye view (top-down):** x vs y, colored by height z. This is the most readable **single-frame** view of the scene layout for non-experts.

`[lidar3d_00000_view_top.png]` *(or `lidar3d_00000.png` from `lidar_3d_samples/`)*

**Single-scan 3D view** (or orthographic top camera): complements the BEV by showing **3D structure** of one LiDAR sweep. Use one representative frame.

`[pointcloud_timeline_3d.png]` *(optional; explain in caption)*

**Multi-frame overlay:** many consecutive frames in one plot, colored by **time index**. This is **not** a single snapshot; static structure stacks while moving objects may **smear**. Include only if you add one sentence in the caption stating that.

---

## 10. Code excerpts and brief explanations

The following snippets are **representative** of our pipeline (see repository for full files). They show how raw **metadata** becomes **supervised windows**, how the **baseline** maps past motion to future points, and how we will **score** predictions.

### 10.1 Building supervised trajectory windows (`waymo_windows.py`)

**Purpose:** Group frames by **sequence**, require the same **object ID** across **past** and **future** indices, and express positions in a **local frame** (origin at the last past point) for stable training.

```python
# code/data_pipeline/waymo_windows.py (excerpt: core loop)
def build_xy_windows(infos_path, past_len=10, future_len=20, max_windows=20000):
    infos = load_infos(infos_path)
    grouped = _group_by_sequence(infos)
    xs, ys = [], []
    for _, frames in grouped.items():
        ids = sorted(frames.keys())
        id_set = set(ids)
        obj_xy = {fid: _xy_by_obj(frames[fid]) for fid in ids}
        for anchor in ids:
            past_ids = [anchor - k for k in range(past_len - 1, -1, -1)]
            fut_ids = [anchor + k for k in range(1, future_len + 1)]
            if not all(fid in id_set for fid in past_ids + fut_ids):
                continue
            common = None
            for fid in past_ids + fut_ids:
                ids_here = set(obj_xy[fid].keys())
                common = ids_here if common is None else (common & ids_here)
                if not common:
                    break
            if not common:
                continue
            for obj_id in common:
                x_abs = np.stack([obj_xy[fid][obj_id] for fid in past_ids], axis=0)
                y_abs = np.stack([obj_xy[fid][obj_id] for fid in fut_ids], axis=0)
                origin = x_abs[-1].copy()
                xs.append((x_abs - origin).astype(np.float32))
                ys.append((y_abs - origin).astype(np.float32))
                if len(xs) >= max_windows:
                    return np.stack(xs), np.stack(ys)
    return np.stack(xs), np.stack(ys)
```

**Explanation:** `past` and `future` are aligned in time; **subtraction by `origin`** centers the target at the last observed step. Outputs are **numpy** arrays shaped for batching into PyTorch tensors.

### 10.2 Baseline LSTM (`lstm_baseline.py`)

**Purpose:** Encode the **past sequence** with an **LSTM**, then regress **all future (x, y)** in one forward pass (shape `[batch, future_len, 2]`).

```python
# code/models/lstm_baseline.py (full module is short)
class LSTMBaseline(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, future_len=20):
        super().__init__()
        self.future_len = future_len
        self.encoder = nn.LSTM(in_dim, hidden_dim, batch_first=True, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_len * 2),
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = h[-1]
        out = self.head(z)
        return out.view(x.shape[0], self.future_len, 2)
```

**Explanation:** `h[-1]` is the **last hidden state** of the last time step (encoding the past). The **head** expands to `future_len * 2` values and reshapes to **future trajectories**.

### 10.3 Evaluation metrics (`metrics.py`)

**Purpose:** Implement **ADE** (mean error over all future steps) and **FDE** (error at the **final** predicted time), matching standard trajectory forecasting practice.

```python
# code/evaluation/metrics.py
def ade(pred, target):  # pred, target: [N, T, 2]
    err = np.linalg.norm(pred - target, axis=-1)
    return float(np.mean(err))

def fde(pred, target):
    err = np.linalg.norm(pred[:, -1, :] - target[:, -1, :], axis=-1)
    return float(np.mean(err))
```

**Explanation:** `ade` is sensitive to **overall** path error; `fde` stresses **endpoint** accuracy (important for short-term vs long-term behavior).

---

## 11. References (informal)

- Waymo Open Dataset - Motion / perception documentation: [https://waymo.com/open/](https://waymo.com/open/)  
- Course project proposal (team): *Trajectory Prediction for Autonomous Driving* - baseline -> multi-agent -> map-aware roadmap.

---

*End of report.*
