"""
Simple EDA for preprocessed Waymo artifacts.

This script reads a Waymo *_infos_*.pkl file and produces:
1) Console summary stats
2) CSV summaries
3) Basic plots in an output directory
"""

import argparse
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def _eda_plot_style():
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=1.05,
        rc={
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description="EDA for Waymo processed info pickle")
    parser.add_argument(
        "--infos",
        type=str,
        default="/scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0_infos_train.pkl",
        help="Path to *_infos_*.pkl file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="How many frames to scan (use -1 for all)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/eda",
        help="Output directory for csv/plots",
    )
    return parser.parse_args()


def safe_len(x):
    return len(x) if hasattr(x, "__len__") else 0


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.infos, "rb") as f:
        infos = pickle.load(f)

    total_frames = len(infos)
    scan_n = total_frames if args.max_samples == -1 else min(args.max_samples, total_frames)
    subset = infos[:scan_n]

    class_counts = Counter()
    points_per_obj = []
    speed_norm = []
    boxes_per_frame = []
    frame_rows = []

    for i, frame in enumerate(tqdm(subset, desc="Scanning frames")):
        ann = frame.get("annos", {})

        names = ann.get("name", np.array([]))
        num_points = ann.get("num_points_in_gt", np.array([]))
        speed_global = ann.get("speed_global", np.zeros((safe_len(names), 2)))
        gt_boxes = ann.get("gt_boxes_lidar", np.zeros((safe_len(names), 9)))

        names = np.array(names)
        num_points = np.array(num_points)
        speed_global = np.array(speed_global)
        gt_boxes = np.array(gt_boxes)

        boxes_per_frame.append(len(names))
        class_counts.update([str(x) for x in names.tolist()])

        if num_points.size > 0:
            points_per_obj.extend(num_points.tolist())

        if speed_global.size > 0 and speed_global.ndim == 2 and speed_global.shape[1] >= 2:
            v = np.linalg.norm(speed_global[:, :2], axis=1)
            speed_norm.extend(v.tolist())

        frame_rows.append(
            {
                "frame_idx": i,
                "frame_id": frame.get("frame_id", f"frame_{i}"),
                "num_objects": len(names),
                "num_gt_boxes": len(gt_boxes),
            }
        )

    # Dataframes and CSVs
    df_frames = pd.DataFrame(frame_rows)
    df_class = pd.DataFrame(
        [{"class_name": k, "count": v} for k, v in class_counts.items()]
    ).sort_values("count", ascending=False)
    df_frames.to_csv(os.path.join(args.out_dir, "frames_summary.csv"), index=False)
    df_class.to_csv(os.path.join(args.out_dir, "class_counts.csv"), index=False)

    # Console summary
    print("\n=== EDA SUMMARY ===")
    print(f"Infos file: {args.infos}")
    print(f"Total frames in file: {total_frames}")
    print(f"Frames scanned: {scan_n}")
    print(f"Avg objects/frame: {np.mean(boxes_per_frame):.2f}" if boxes_per_frame else "Avg objects/frame: N/A")
    print(f"Median objects/frame: {np.median(boxes_per_frame):.2f}" if boxes_per_frame else "Median objects/frame: N/A")
    if speed_norm:
        print(f"Avg speed norm (m/s): {np.mean(speed_norm):.2f}")
        print(f"P95 speed norm (m/s): {np.percentile(speed_norm, 95):.2f}")
    else:
        print("Speed norm stats: N/A")
    print("\nTop classes:")
    if not df_class.empty:
        print(df_class.head(10).to_string(index=False))
    else:
        print("No class labels found.")

    # Plots (seaborn + matplotlib)
    _eda_plot_style()
    hist_color = sns.color_palette("deep")[0]

    if not df_class.empty:
        top = df_class.head(10).copy()
        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
        sns.barplot(
            data=top,
            x="class_name",
            y="count",
            order=top["class_name"].tolist(),
            ax=ax,
            palette=sns.color_palette("viridis", n_colors=len(top)),
            edgecolor=".2",
            linewidth=0.6,
        )
        ax.set_title("Top-10 class counts")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        sns.despine(ax=ax, left=False, bottom=False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "class_counts_top10.png"), dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    if boxes_per_frame:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        sns.histplot(boxes_per_frame, bins=30, color=hist_color, ax=ax, edgecolor="white", linewidth=0.4)
        ax.set_title("Objects per frame")
        ax.set_xlabel("Object count")
        ax.set_ylabel("Frequency")
        sns.despine(ax=ax, left=False, bottom=False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "objects_per_frame_hist.png"), dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    if points_per_obj:
        clipped = np.clip(np.array(points_per_obj), 0, np.percentile(points_per_obj, 99))
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        sns.histplot(clipped, bins=40, color=hist_color, ax=ax, kde=False, edgecolor="white", linewidth=0.35)
        ax.set_title("Num points per GT object (clipped @ p99)")
        ax.set_xlabel("num_points_in_gt")
        ax.set_ylabel("Frequency")
        sns.despine(ax=ax, left=False, bottom=False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "num_points_per_object_hist.png"), dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    if speed_norm:
        clipped = np.clip(np.array(speed_norm), 0, np.percentile(speed_norm, 99))
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        sns.histplot(clipped, bins=40, color=hist_color, ax=ax, kde=False, edgecolor="white", linewidth=0.35)
        ax.set_title("Object speed norm (m/s, clipped @ p99)")
        ax.set_xlabel("Speed magnitude")
        ax.set_ylabel("Frequency")
        sns.despine(ax=ax, left=False, bottom=False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "speed_norm_hist.png"), dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"\nSaved outputs to: {args.out_dir}")
    print(" - frames_summary.csv")
    print(" - class_counts.csv")
    print(" - class_counts_top10.png")
    print(" - objects_per_frame_hist.png")
    print(" - num_points_per_object_hist.png")
    print(" - speed_norm_hist.png")


if __name__ == "__main__":
    main()
