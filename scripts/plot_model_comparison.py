#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def load_history(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != "epoch" else int(v) for k, v in row.items()})
    return rows


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser("Plot model comparison (loss + ADE/FDE)")
    p.add_argument("--run", action="append", nargs=3, metavar=("NAME", "HISTORY_CSV", "METRICS_JSON"), required=True)
    p.add_argument("--out-dir", type=str, default="results/comparison")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = []
    for name, hist_path, metrics_path in args.run:
        runs.append(
            {
                "name": name,
                "history": load_history(hist_path),
                "metrics": load_json(metrics_path),
            }
        )

    plt.figure(figsize=(8, 5))
    for r in runs:
        epochs = [row["epoch"] for row in r["history"]]
        val_loss = [row["val_loss"] for row in r["history"]]
        plt.plot(epochs, val_loss, label=r["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation Loss by Model")
    plt.legend()
    plt.tight_layout()
    loss_png = os.path.join(args.out_dir, "val_loss_comparison.png")
    plt.savefig(loss_png, dpi=160)
    plt.close()

    labels = [r["name"] for r in runs]
    ades = [r["metrics"]["ADE"] for r in runs]
    fdes = [r["metrics"]["FDE"] for r in runs]

    x = list(range(len(labels)))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], ades, width=width, label="ADE")
    plt.bar([i + width / 2 for i in x], fdes, width=width, label="FDE")
    plt.xticks(x, labels)
    plt.ylabel("Error (meters)")
    plt.title("ADE/FDE Comparison")
    plt.legend()
    plt.tight_layout()
    bar_png = os.path.join(args.out_dir, "ade_fde_comparison.png")
    plt.savefig(bar_png, dpi=160)
    plt.close()

    summary = []
    for r in runs:
        summary.append(
            {
                "model": r["name"],
                "ADE": r["metrics"]["ADE"],
                "FDE": r["metrics"]["FDE"],
                "samples": r["metrics"].get("samples", -1),
            }
        )
    summary_csv = os.path.join(args.out_dir, "metrics_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "ADE", "FDE", "samples"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"saved: {loss_png}")
    print(f"saved: {bar_png}")
    print(f"saved: {summary_csv}")


if __name__ == "__main__":
    main()
