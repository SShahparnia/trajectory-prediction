# Trajectory Prediction Project - Slide Template (Visual-First)

Use this as a direct script for your deck.  
Each slide has: **Title**, **Visual(s)**, **Talking points**.

---

## Slide 1 - Title
**Title:** Trajectory Prediction for Autonomous Driving  
**Subtitle:** LSTM vs Transformer vs Multi-Agent LSTM on Waymo (CMPE 249/257 Project)  
**Team:** Shervan Shahparnia, Aaron Sam, Bhavdeep Randhawa, Atilla Sayin

**Visual(s):**
- Full-width hero image: `results/lidar_sim/demo1_3d_clean/lidar_bev_sim.gif` (or one clean frame)

**Talking points:**
- We predict future 2D trajectories from past motion.
- We compare single-agent and multi-agent models.
- Focus is practical, reproducible results under HPC constraints.

---

## Slide 2 - Overview / Agenda
**Title:** Overview

**Visual(s):**
- Simple timeline graphic (5 blocks): Problem -> Data -> Models -> Results -> Demo/Takeaways

**Talking points:**
- Problem and motivation
- Dataset and preprocessing
- Model variants (LSTM, Transformer, Multi-Agent LSTM)
- Quantitative + qualitative evaluation
- Limitations and next steps

---

## Slide 3 - Motivation
**Title:** Why Trajectory Prediction Matters

**Visual(s):**
- 1 BEV point cloud frame with moving agents highlighted

**Talking points:**
- Better trajectory forecasts support safer planning.
- Key challenge: dense traffic interactions and uncertainty.
- Goal: reduce future position error (ADE/FDE).

---

## Slide 4 - Problem Formulation
**Title:** Problem Setup (Input -> Output)

**Visual(s):**
- Diagram: past window (P=10) -> model -> future window (F=20)
- Optional equation: f(past[, neighbors]) = future

**Talking points:**
- Input: past XY trajectory (and neighbors for multi-agent).
- Output: future XY trajectory in local frame.
- We model BEV motion (XY), not full 3D XYZ forecasting.

---

## Slide 5 - Dataset Source
**Title:** Dataset and Data Source

**Visual(s):**
- Folder/data flow diagram from `/scratch/lts-data/cmpe249-fa22/Waymo132` -> `infos_*.pkl` -> windows

**Talking points:**
- We use course-hosted Waymo processed metadata and LiDAR.
- Raw data is large; project pipeline uses processed `infos` pickles.
- Train/val/test split is sequence-based to avoid leakage.

---

## Slide 6 - Data Splits
**Title:** Train / Validation / Test Split

**Visual(s):**
- Pie or bar chart for split counts (train, val, test)
- Optional text from `data/manifest.json` stats

**Talking points:**
- Combined frames: 4761
- Sequence-level split (~70/15/15)
- Ensures same segment does not appear across splits.

---

## Slide 7 - EDA: Class Imbalance
**Title:** EDA - Object Distribution

**Visual(s):**
- `results/eda_full/class_counts_top10.png`

**Talking points:**
- Strong class imbalance (vehicles dominate).
- Important for fairness and generalization discussion.

---

## Slide 8 - EDA: Scene Density
**Title:** EDA - Objects / Frame and Point Density

**Visual(s):**
- `results/eda_smoke/objects_per_frame_hist.png`
- `results/eda_smoke/num_points_per_object_hist.png`

**Talking points:**
- Frame complexity varies significantly.
- Variable density motivates robust, context-aware modeling.

---

## Slide 9 - LiDAR Context Visualization
**Title:** LiDAR Context (BEV/3D)

**Visual(s):**
- `results/lidar_sim/demo1/lidar_bev_sim.gif`
- Optionally show clean 3D gif next to it

**Talking points:**
- This is data playback (scene context), not prediction.
- Useful for explaining environment structure and object motion.

---

## Slide 10 - Baseline Model
**Title:** Single-Agent LSTM Baseline

**Visual(s):**
- Architecture block diagram (LSTM encoder + MLP head)

**Talking points:**
- Encodes ego past trajectory only.
- Strong baseline for motion continuity patterns.

---

## Slide 11 - Transformer Model
**Title:** Single-Agent Transformer Baseline

**Visual(s):**
- Encoder diagram: input projection + positional encoding + transformer encoder

**Talking points:**
- Captures temporal relationships differently from LSTM.
- Same I/O setup for fair comparison.

---

## Slide 12 - Multi-Agent Model
**Title:** Multi-Agent LSTM (Interaction-Aware)

**Visual(s):**
- Diagram: ego encoder + neighbor encoder + mask/pooling + prediction head
- Show K-neighbor padding/mask concept

**Talking points:**
- Uses nearby agents (max neighbors = 12).
- Neighbors encoded and pooled with masks.
- Adds interaction signal beyond ego-only history.

---

## Slide 13 - Training and Evaluation Protocol
**Title:** Experimental Setup

**Visual(s):**
- Table: common hyperparameters, windows, epochs, metrics

**Talking points:**
- Same split protocol across models.
- Metrics: ADE and FDE on held-out test set.
- CPU-friendly pipeline for reproducibility on cluster.

---

## Slide 14 - Training Curves
**Title:** Validation Loss Comparison

**Visual(s):**
- `results/experiments/<run_tag>/comparison/val_loss_comparison.png`

**Talking points:**
- All models learn over epochs.
- Compare convergence behavior and stability.

---

## Slide 15 - Main Metrics
**Title:** ADE / FDE Comparison

**Visual(s):**
- `results/experiments/<run_tag>/comparison/ade_fde_comparison.png`
- Tiny table from `metrics_summary.csv`

**Talking points:**
- Lower is better for both metrics.
- Present best model and practical significance.

---

## Slide 16 - Qualitative Rollout (LSTM)
**Title:** Model Rollout - LSTM

**Visual(s):**
- `results/model_rollout_sim/overhead_lstm/model_rollout_lstm_sample0163.gif`

**Talking points:**
- Black: past input
- Green: ground-truth future (`future_gt`)
- Red dashed: model prediction (`future_pred`)

---

## Slide 17 - Qualitative Rollout (Transformer)
**Title:** Model Rollout - Transformer

**Visual(s):**
- `results/model_rollout_sim/overhead_transformer/model_rollout_transformer_sample0163.gif`

**Talking points:**
- Compare path smoothness and endpoint behavior.
- Discuss where it matches or diverges from GT.

---

## Slide 18 - Qualitative Rollout (Multi-Agent)
**Title:** Model Rollout - Multi-Agent LSTM

**Visual(s):**
- `results/model_rollout_sim/overhead_multi_lstm/model_rollout_multi_lstm_sample0163.gif`

**Talking points:**
- Includes neighbor context during prediction.
- Explain interaction-aware behavior in the scene.

---

## Slide 19 - Side-by-Side Qualitative Comparison
**Title:** Same Scene, Three Models

**Visual(s):**
- 3-column layout:
  - LSTM gif
  - Transformer gif
  - Multi-agent gif

**Talking points:**
- Same sample index for fair qualitative comparison.
- Highlight failure modes and best behavior.

---

## Slide 20 - What We Learned
**Title:** Key Findings

**Visual(s):**
- 3 concise callout boxes with icons (Data, Model, Result)

**Talking points:**
- Forecasting from processed infos is feasible and reproducible.
- Multi-agent context can help but may vary by scenario.
- Visual + metric agreement strengthens conclusions.

---

## Slide 21 - Limitations
**Title:** Current Limitations

**Visual(s):**
- Risk/limitation table

**Talking points:**
- XY-only forecasting (no explicit Z/map constraints).
- Simplified neighbor pooling in multi-agent model.
- No full raw Motion TFRecord pipeline in this milestone.
- Hardware/runtime constraints limited broader sweeps.

---

## Slide 22 - Future Work
**Title:** Next Steps

**Visual(s):**
- Roadmap timeline (short-term -> final paper)

**Talking points:**
- Map-aware branch (lane/drivable constraints)
- Stronger interaction blocks (attention/graph models)
- Per-class / scenario-specific evaluation
- Optional raw parser + richer simulation tooling

---

## Slide 23 - Reproducibility
**Title:** Reproducibility and Pipeline

**Visual(s):**
- Command snippet box:
  - `bash scripts/run_friday_pipeline.sh`
  - `sbatch slurm/run_friday_pipeline.sbatch`

**Talking points:**
- Single command for train/eval/plots.
- Fixed splits and saved artifacts for repeatability.

---

## Slide 24 - References
**Title:** References

**Visual(s):**
- Clean numbered list

**Talking points:**
- Waymo Open Dataset
- Key trajectory forecasting references you used
- Any code/documentation references for tooling

---

## Slide 25 - Q&A
**Title:** Thank You

**Visual(s):**
- Best-looking GIF frame + contact/team info

**Talking points:**
- Invite questions on model design, evaluation, and next steps.

---

## Optional Appendix Slides
1. Hyperparameter table per model  
2. Additional rollout examples (best/worst cases)  
3. Runtime and resource usage table  
4. Data preprocessing details (window construction and normalization)

