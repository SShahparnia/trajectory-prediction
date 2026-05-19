#!/usr/bin/env python3
"""
Train traditional ML models (Random Forest, XGBoost) on trajectory prediction
using the real Waymo data pipeline.
"""

import sys
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from traj_code.data_pipeline.waymo_windows import build_xy_windows
from traj_code.evaluation.metrics import ade, fde

PAST_LEN   = 10
FUTURE_LEN = 20
DATA_DIR   = Path("data")
OUT_BASE   = Path("results")
OUT_BASE.mkdir(exist_ok=True)


def engineer_features(x: np.ndarray) -> np.ndarray:
    """x: [N, T, 2] -> [N, features] with position, velocity, acceleration, speed."""
    N = x.shape[0]
    pos   = x.reshape(N, -1)                                   # [N, T*2]
    vel   = np.diff(x, axis=1).reshape(N, -1)                  # [N, (T-1)*2]
    acc   = np.diff(x, n=2, axis=1).reshape(N, -1)             # [N, (T-2)*2]
    speed = np.linalg.norm(np.diff(x, axis=1), axis=-1)        # [N, T-1]
    path_len = speed.sum(axis=1, keepdims=True)                 # [N, 1]
    net_disp = (x[:, -1] - x[:, 0])                            # [N, 2]
    return np.concatenate([pos, vel, acc, speed, path_len, net_disp], axis=1)


def evaluate_model(model, Xf, y, label: str):
    pred = model.predict(Xf).reshape(-1, FUTURE_LEN, 2)
    a = ade(pred, y)
    f = fde(pred, y)
    print(f"  {label}: ADE={a:.4f}m  FDE={f:.4f}m")
    return a, f


def main():
    print("=" * 70)
    print("TRAINING TRADITIONAL ML MODELS FOR TRAJECTORY PREDICTION")
    print("=" * 70)

    # ── Load real Waymo data ───────────────────────────────────────────────
    print("\nBuilding trajectory windows from Waymo data...")
    X_train, y_train = build_xy_windows(
        str(DATA_DIR / "infos_train.pkl"), PAST_LEN, FUTURE_LEN, max_windows=5000)
    X_val, y_val = build_xy_windows(
        str(DATA_DIR / "infos_val.pkl"),   PAST_LEN, FUTURE_LEN, max_windows=1000)
    X_test, y_test = build_xy_windows(
        str(DATA_DIR / "infos_test.pkl"),  PAST_LEN, FUTURE_LEN, max_windows=2000)
    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # ── Feature engineering ────────────────────────────────────────────────
    Xf_train = engineer_features(X_train)
    Xf_val   = engineer_features(X_val)
    Xf_test  = engineer_features(X_test)
    yf_train = y_train.reshape(y_train.shape[0], -1)   # [N, 40]
    print(f"  Feature dim: {Xf_train.shape[1]}")

    results = {}

    # ── Random Forest ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training Random Forest...")
    rf_out = OUT_BASE / "train_random_forest"
    rf_out.mkdir(exist_ok=True)

    # RandomForestRegressor natively supports multi-output
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=5,
        n_jobs=-1, random_state=42
    )
    rf_model.fit(Xf_train, yf_train)
    print("  Evaluating...")
    val_ade,  val_fde  = evaluate_model(rf_model, Xf_val,  y_val,  "Val ")
    test_ade, test_fde = evaluate_model(rf_model, Xf_test, y_test, "Test")

    joblib.dump(rf_model, rf_out / "best_model.pkl")
    res = dict(val_ade=val_ade, val_fde=val_fde, test_ade=test_ade, test_fde=test_fde)
    with open(rf_out / "test_results.json", "w") as f:
        json.dump({k: float(v) for k, v in res.items()}, f, indent=2)
    results["Random Forest"] = res
    print(f"  Saved to {rf_out}")

    # ── XGBoost ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training XGBoost...")
    xgb_out = OUT_BASE / "train_xgboost"
    xgb_out.mkdir(exist_ok=True)

    # MultiOutputRegressor wraps XGBoost for multi-output regression
    xgb_model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1, random_state=42, verbosity=0
        ),
        n_jobs=1   # XGBoost is already parallel; avoid thread contention
    )
    xgb_model.fit(Xf_train, yf_train)
    print("  Evaluating...")
    val_ade,  val_fde  = evaluate_model(xgb_model, Xf_val,  y_val,  "Val ")
    test_ade, test_fde = evaluate_model(xgb_model, Xf_test, y_test, "Test")

    joblib.dump(xgb_model, xgb_out / "best_model.pkl")
    res = dict(val_ade=val_ade, val_fde=val_fde, test_ade=test_ade, test_fde=test_fde)
    with open(xgb_out / "test_results.json", "w") as f:
        json.dump({k: float(v) for k, v in res.items()}, f, indent=2)
    results["XGBoost"] = res
    print(f"  Saved to {xgb_out}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:20s}  Test ADE={r['test_ade']:.4f}m  Test FDE={r['test_fde']:.4f}m")

    with open(OUT_BASE / "traditional_results.json", "w") as f:
        json.dump({k: {m: float(v) for m, v in r.items()} for k, r in results.items()}, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
