# Slurm Pipeline (Train/Val/Test)

This folder contains a ready-to-run Stage 1 pipeline aligned with your proposal:

- **Train** baseline LSTM on `train` split
- **Validate** on `val` split
- **Test** on `test` split

By default, it now uses **direct scratch infos paths** from
`/scratch/lts-data/cmpe249-fa22/Waymo132/..._infos_{train,val}.pkl`,
so it still runs even if local `data/` split files are missing.

## Files

- `train_stage1_lstm.sbatch` - training job (GPU)
- `eval_checkpoint.sbatch` - evaluation job (CPU/GPU optional; CPU by default)
- `submit_stage1_pipeline.sh` - submits train -> val/test jobs with dependencies
- `pipeline_defaults.env` - shared defaults used by both submit and train scripts

## Quick start

From repo root:

```bash
chmod +x slurm/submit_stage1_pipeline.sh
chmod +x slurm/train_stage1_lstm.sbatch
chmod +x slurm/eval_checkpoint.sbatch
./slurm/submit_stage1_pipeline.sh
```

This creates an experiment folder like:

- `results/experiments/stage1_YYYYMMDD_HHMMSS/`
  - `best_model.pt`
  - `train_history.csv`
  - `run_info.json`
  - `metrics_val.json`
  - `metrics_test.json`

## Common overrides

You can change defaults in `pipeline_defaults.env` once, or override per-run:

```bash
RUN_TAG=stage1_full_gpu \
EPOCHS=30 \
BATCH_SIZE=256 \
MAX_WINDOWS=200000 \
VAL_MAX_WINDOWS=50000 \
PAST_LEN=10 \
FUTURE_LEN=20 \
./slurm/submit_stage1_pipeline.sh
```

To point at different splits:

```bash
TRAIN_INFOS=data/infos_train.pkl \
VAL_INFOS=data/infos_val.pkl \
TEST_INFOS=data/infos_test.pkl \
./slurm/submit_stage1_pipeline.sh
```

If you later recreate local split pickles (`data/infos_train.pkl`, etc.), just override
`TRAIN_INFOS`, `VAL_INFOS`, `TEST_INFOS` or update `slurm/pipeline_defaults.env`.

## Notes

- If your cluster requires an account/partition/QOS, add `#SBATCH` directives in the `.sbatch` files.
- The evaluation script now evaluates on the full provided `--infos` set (no random 80/20 split).
- For final reported numbers, avoid tiny debug `MAX_WINDOWS`.
