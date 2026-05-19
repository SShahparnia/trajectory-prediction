#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def _plot_style():
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.85,
        rc={
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "grid.alpha": 0.45,
            "grid.linestyle": "--",
        },
    )


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

    _plot_style()
    palette = sns.color_palette("deep", n_colors=max(3, len(runs)))

    # --- Validation loss (long-form for seaborn)
    loss_rows = []
    for i, r in enumerate(runs):
        for row in r["history"]:
            loss_rows.append(
                {
                    "epoch": row["epoch"],
                    "val_loss": row["val_loss"],
                    "model": r["name"],
                    "color_idx": i,
                }
            )
    df_loss = pd.DataFrame(loss_rows)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    sns.lineplot(
        data=df_loss,
        x="epoch",
        y="val_loss",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        ax=ax,
        palette=palette[: len(runs)],
        linewidth=2.2,
        markersize=6,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss by model")
    ax.legend(title=None, frameon=True, framealpha=0.95, loc="best")
    sns.despine(ax=ax, left=False, bottom=False)
    plt.tight_layout()
    loss_png = os.path.join(args.out_dir, "val_loss_comparison.png")
    fig.savefig(loss_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- ADE / FDE grouped bars
    labels = [r["name"] for r in runs]
    ades = [r["metrics"]["ADE"] for r in runs]
    fdes = [r["metrics"]["FDE"] for r in runs]
    df_bar = pd.DataFrame(
        {
            "model": labels,
            "ADE": ades,
            "FDE": fdes,
        }
    )
    df_long = df_bar.melt(id_vars="model", var_name="metric", value_name="meters")

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)
    sns.barplot(
        data=df_long,
        x="model",
        y="meters",
        hue="metric",
        ax=ax,
        palette={"ADE": "#4c72b0", "FDE": "#dd8452"},
        edgecolor=".25",
        linewidth=0.8,
        saturation=0.92,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Error (meters)")
    ax.set_title("ADE / FDE on test windows")
    ax.legend(title=None, frameon=True, framealpha=0.95)
    try:
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", padding=2, fontsize=9)
    except AttributeError:
        pass
    sns.despine(ax=ax, left=False, bottom=False)
    plt.tight_layout()
    bar_png = os.path.join(args.out_dir, "ade_fde_comparison.png")
    fig.savefig(bar_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

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
