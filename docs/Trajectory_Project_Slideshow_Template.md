# Trajectory Prediction Project - Slide Template (Visual-First)

Use this as a direct script for your deck.  
Each slide has: **Title**, **Visual(s)**, **Talking points**.

**Bundled visuals for slides 5–25:** `docs/presentation_assets/` (full list: `docs/presentation_assets/README_SLIDES.txt`). Slides 3–4 share that folder too. Regenerate everything:

```bash
python scripts/generate_slide03_slide04_visuals.py
python scripts/generate_presentation_assets_slides05_plus.py
```

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
- `docs/presentation_assets/slide03_motivation_bev.png` (BEV-style schematic: pseudo-LiDAR ground, highlighted agents, past trajectory + plausible futures)
- Regenerate: `python scripts/generate_slide03_slide04_visuals.py`

**Talking points:**
- Better trajectory forecasts help planners anticipate where nearby vehicles and pedestrians will likely move over the next few seconds, which supports safer lane changes, braking, and collision avoidance.
- A core challenge is that real traffic is highly interactive and uncertain: agents influence each other, scenes are dense, and the same current state can still lead to multiple plausible futures.
- Our modeling objective is to reduce forecast error over a prediction horizon, measured by ADE (average displacement error across all future steps) and FDE (final-step displacement error at the endpoint).

---

## Slide 4 - Problem Formulation
**Title:** Problem Setup (Input -> Output)

**Visual(s):**
- `docs/presentation_assets/slide04_problem_setup.png` (pipeline: \(X_{t-P+1:t}\) → \(f_\theta\) → \(\hat{X}_{t+1:t+F}\), multi-agent note)
- Regenerate: `python scripts/generate_slide03_slide04_visuals.py`

**Talking points:**
- Input: past XY trajectory (and neighbors for multi-agent).
- Output: future XY trajectory in local frame.
- We model BEV motion (XY), not full 3D XYZ forecasting.

---

## Slide 5 - Dataset Source
**Title:** Dataset and Data Source

**Visual(s):**
- `docs/presentation_assets/slide05_dataset_flow.png`

**Talking points:**
- Raw Waymo-related assets live on HPC; our repo implements parsing, trajectory extraction, and windowing.
- We generate sequence-stratified train/val/test splits ourselves (no shared segments across splits).
- Everything downstream—datasets, training, metrics—is produced by our scripts and checkpoints.

---

## Slide 6 - Data Splits
**Title:** Train / Validation / Test Split

**Visual(s):**
- `docs/presentation_assets/slide06_train_val_test_splits.png`

**Talking points:**
- Combined frames: 4761
- Sequence-level split (~70/15/15)
- Ensures same segment does not appear across splits.

---

## Slide 7 - EDA: Class Imbalance
**Title:** EDA - Object Distribution

**Visual(s):**
- `docs/presentation_assets/slide07_eda_class_counts_top10.png`

**Talking points:**
- Strong class imbalance (vehicles dominate).
- Important for fairness and generalization discussion.

---

## Slide 8 - EDA: Scene Density
**Title:** EDA - Objects / Frame and Point Density

**Visual(s):**
- `docs/presentation_assets/slide08_eda_density_panel.png` (or separate `slide08a_*` / `slide08b_*` if Pillow missing during generation)

**Talking points:**
- Frame complexity varies significantly.
- Variable density motivates robust, context-aware modeling.

---

## Slide 9 - LiDAR Context Visualization
**Title:** LiDAR Context (BEV/3D)

**Visual(s):**
- `docs/presentation_assets/slide09_lidar_bev_clean.gif`
- `docs/presentation_assets/slide09_lidar_3d_clean.gif`

**Talking points:**
- This is data playback (scene context), not prediction.
- Useful for explaining environment structure and object motion.

---

## Slide 10 - Baseline Model
**Title:** Single-Agent LSTM Baseline

**Visual(s):**
- `docs/presentation_assets/slide10_architecture_lstm.png`

**Talking points:**
- Encodes ego past trajectory only.
- Strong baseline for motion continuity patterns.

---

## Slide 11 - Transformer Model
**Title:** Single-Agent Transformer Baseline

**Visual(s):**
- `docs/presentation_assets/slide11_architecture_transformer.png`

**Talking points:**
- Captures temporal relationships differently from LSTM.
- Same I/O setup for fair comparison.

---

## Slide 12 - Multi-Agent Model
**Title:** Multi-Agent LSTM (Interaction-Aware)

**Visual(s):**
- `docs/presentation_assets/slide12_architecture_multi_agent_lstm.png`

**Talking points:**
- Uses nearby agents (max neighbors = 12).
- Neighbors encoded and pooled with masks.
- Adds interaction signal beyond ego-only history.

---

## Slide 13 - Training and Evaluation Protocol
**Title:** Experimental Setup

**Visual(s):**
- `docs/presentation_assets/slide13_experimental_setup_table.png`

**Talking points:**
- Same split protocol across models.
- Metrics: ADE and FDE on held-out test set.
- CPU-friendly pipeline for reproducibility on cluster.

---

## Slide 14 - Training Curves
**Title:** Validation Loss Comparison

**Visual(s):**
- `docs/presentation_assets/slide14_val_loss_comparison.png`

**Talking points:**
- All models learn over epochs.
- Compare convergence behavior and stability.

---

## Slide 15 - Main Metrics
**Title:** ADE / FDE Comparison

**Visual(s):**
- `docs/presentation_assets/slide15_ade_fde_comparison.png`
- `docs/presentation_assets/slide15_metrics_table.png` + `slide15_metrics_summary.csv`

**Talking points:**
- Lower is better for both metrics.
- Present best model and practical significance.

---

## Slide 16 - Qualitative Rollout (LSTM)
**Title:** Model Rollout - LSTM

**Visual(s):**
- `docs/presentation_assets/slide16_rollout_lstm.gif`

**Talking points:**
- Black: past input
- Green: ground-truth future (`future_gt`)
- Red dashed: model prediction (`future_pred`)

---

## Slide 17 - Qualitative Rollout (Transformer)
**Title:** Model Rollout - Transformer

**Visual(s):**
- `docs/presentation_assets/slide17_rollout_transformer.gif`

**Talking points:**
- Compare path smoothness and endpoint behavior.
- Discuss where it matches or diverges from GT.

---

## Slide 18 - Qualitative Rollout (Multi-Agent)
**Title:** Model Rollout - Multi-Agent LSTM

**Visual(s):**
- `docs/presentation_assets/slide18_rollout_multi_lstm.gif`

**Talking points:**
- Includes neighbor context during prediction.
- Explain interaction-aware behavior in the scene.

---

## Slide 19 - Side-by-Side Qualitative Comparison
**Title:** Same Scene, Three Models

**Visual(s):**
- `docs/presentation_assets/slide19_three_models_triptych.png` (first frame of each rollout GIF)

**Talking points:**
- Same sample index for fair qualitative comparison.
- Highlight failure modes and best behavior.

---

## Slide 20 - What We Learned
**Title:** Key Findings

**Visual(s):**
- `docs/presentation_assets/slide20_key_findings.png`

**Talking points:**
- Forecasting from processed infos is feasible and reproducible.
- Multi-agent context can help but may vary by scenario.
- Visual + metric agreement strengthens conclusions.

---

## Slide 21 - Limitations
**Title:** Current Limitations

**Visual(s):**
- `docs/presentation_assets/slide21_limitations_table.png`

**Talking points:**
- XY-only forecasting (no explicit Z/map constraints).
- Simplified neighbor pooling in multi-agent model.
- No full raw Motion TFRecord pipeline in this milestone.
- Hardware/runtime constraints limited broader sweeps.

---

## Slide 22 - Future Work
**Title:** Next Steps

**Visual(s):**
- `docs/presentation_assets/slide22_future_work_roadmap.png`

**Talking points:**
- Map-aware branch (lane/drivable constraints)
- Stronger interaction blocks (attention/graph models)
- Per-class / scenario-specific evaluation
- Optional raw parser + richer simulation tooling

---

## Slide 23 - Reproducibility
**Title:** Reproducibility and Pipeline

**Visual(s):**
- `docs/presentation_assets/slide23_reproducibility_commands.png`

**Talking points:**
- Single command for train/eval/plots.
- Fixed splits and saved artifacts for repeatability.

---

## Slide 24 - References
**Title:** References

**Visual(s):**
- `docs/presentation_assets/slide24_references.png`

**Talking points:**
- Waymo Open Dataset
- Key trajectory forecasting references you used
- Any code/documentation references for tooling

---

## Slide 25 - Q&A
**Title:** Thank You

**Visual(s):**
- `docs/presentation_assets/slide25_thank_you.png`

**Talking points:**
- Invite questions on model design, evaluation, and next steps.

---

## Optional Appendix Slides
1. Hyperparameter table per model  
2. Additional rollout examples (best/worst cases)  
3. Runtime and resource usage table  
4. Data preprocessing details (window construction and normalization)

