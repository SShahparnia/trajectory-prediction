# 3-Week Execution Plan (Team Roadmap)

This is the project plan for the next ~3 weeks.  
Data is already on HPC; focus is on **training, evaluation, and final delivery**.

## Global Rules (apply every week)

- Official runs use: `data/infos_train.pkl`, `data/infos_val.pkl`, `data/infos_test.pkl`.
- No final metrics from tiny debug subsets (`max_windows` too low).
- Every experiment must log: split, `past_len`, `future_len`, seed, model config, checkpoint path.

---

## Week 1 — Lock Scope + Deliver Strong Stage 1 Baseline

**Main outcome:** reproducible single-agent baseline with defensible metrics and visuals.

### Team priorities

- **Aaron**
  - Confirm split/path sanity for all `infos_*.pkl` (sample path checks on HPC).
  - Freeze canonical train/val/test settings and window lengths.
  - Write one-paragraph scope statement (Motion-aligned vs processed-scope practical path).
- **Shervan**
  - Run baseline training on full `infos_train.pkl` with validation on `infos_val.pkl`.
  - Execute 2-3 controlled runs (seed/lr/hidden size) and save checkpoints/curves.
  - Keep one primary baseline (LSTM), optional Transformer only if time allows.
- **Bhavdeep**
  - Finalize feature conventions for Stage 1 input (local frame assumptions).
  - Prepare feature hooks needed for Stage 2 neighbors (without full integration yet).
- **Atilla**
  - Run ADE/FDE evaluation on held-out split.
  - Produce per-class ADE/FDE where labels allow.
  - Build first qualitative prediction-vs-ground-truth plot set.

### Week 1 deliverables

- `results/experiments/stage1_metrics.csv`
- `results/experiments/stage1_qualitative/`
- `docs/SCOPE_DECISION.md` (or equivalent section in report)
- Updated progress section with Stage 1 numbers

### Week 1 exit criteria

- Stage 1 has reproducible ADE/FDE on official splits.
- Per-class metric table exists (or clear note on label limitations).
- At least 2 qualitative prediction figures are ready for report use.

---

## Week 2 — Implement and Validate Multi-Agent Stage

**Main outcome:** Stage 2 model trained and directly compared to Stage 1.

### Team priorities

- **Aaron**
  - Implement/verify neighbor extraction pipeline from `annos` on same split files.
  - Ensure deterministic data assembly and masking behavior.
- **Shervan**
  - Integrate multi-agent architecture (attention/pooling interaction block).
  - Train Stage 2 with same evaluation protocol as Stage 1.
  - Run at least one ablation (e.g., different K neighbors or no velocity feature).
- **Bhavdeep**
  - Own neighbor feature definitions (KNN/radius, local coordinates, velocity encoding).
  - Tune padding/mask strategy for variable number of agents.
- **Atilla**
  - Evaluate Stage 2 (overall + per-class ADE/FDE).
  - Produce Stage 2 vs Stage 1 comparison plots/tables.
  - Select examples where interactions clearly improve behavior.

### Week 2 deliverables

- `results/experiments/stage2_metrics.csv`
- `results/experiments/stage2_vs_stage1.md`
- Side-by-side qualitative comparison figures

### Week 2 exit criteria

- Stage 2 is trained and evaluated on the same official splits.
- Clear comparison against Stage 1 exists (table + brief interpretation).
- Team decides whether to spend Week 3 on full map-aware integration or fallback.

---

## Week 3 — Map-Aware (If Feasible) + Final Packaging

**Main outcome:** final submission package complete; map-aware included if feasible.

### Team priorities

- **Aaron + Bhavdeep**
  - Decide map source feasibility early in the week:
    - Motion map features (if practical), or
    - LiDAR-BEV geometric prior fallback.
  - If not feasible, write explicit limitation + fallback rationale.
- **Shervan**
  - Integrate map-conditioned branch if feasible and run training/eval.
  - Otherwise stabilize best Stage 2 model and run final test evaluation.
- **Atilla**
  - Final metrics table across stages: Stage 1 / Stage 2 / Stage 3 (or N/A with reason).
  - Final figure set (3-4 strongest visuals + key metric chart).
  - Lead report/slides merge and consistency check.

### Week 3 deliverables

- `results/experiments/final_metrics_table.csv`
- Final report PDF + slide deck
- Reproducibility appendix (commands, seeds, checkpoints, split paths)
- Map-aware output **or** documented deferral note

### Week 3 exit criteria

- Submission materials are complete and internally consistent.
- Claims in report match implemented work.
- Team can reproduce headline numbers from saved checkpoints/configs.

---

## Final “Done” Checklist

```
[ ] Week 1 complete: Stage 1 baseline + per-class metrics + qualitative figures
[ ] Week 2 complete: Stage 2 multi-agent implemented and compared to Stage 1
[ ] Week 3 complete: Stage 3 map-aware done OR explicitly deferred with rationale
[ ] Final report/slides complete with reproducibility details
```

---

## Ownership Summary

- **Aaron:** data/split integrity, scope consistency, pipeline reliability
- **Bhavdeep:** feature engineering, multi-agent features, map-aware feasibility/fallback
- **Shervan:** model implementation + training execution (Stage 1/2/3 where feasible)
- **Atilla:** evaluation tables, figure curation, final report/presentation integration

---

**Last updated:** 2026-04-06
