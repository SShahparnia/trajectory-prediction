#!/usr/bin/env python3
"""
Point-cloud EDA for preprocessed Waymo data.

Uses the processed infos pickle to locate frame-level .npy point clouds,
then produces:
1) CSV with per-frame point count and XYZ ranges
2) Aggregate histograms
3) BEV scatter plots for sampled frames
"""

import argparse
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="EDA on Waymo point clouds")
    parser.add_argument(
        "--infos",
        type=str,
        default="/scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0_infos_train.pkl",
        help="Path to *_infos_*.pkl file",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default="/scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0",
        help="Root dir that contains segment folders with frame .npy files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Number of frames to scan (-1 = all)",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=5,
        help="Number of frames to save BEV plots for",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/pointcloud_eda",
        help="Output directory",
    )
    parser.add_argument(
        "--timeline-frames",
        type=int,
        default=12,
        help="Number of frames to combine in one 3D timeline image",
    )
    parser.add_argument(
        "--timeline-point-cap",
        type=int,
        default=120000,
        help="Max total points used in combined 3D timeline plot",
    )
    parser.add_argument(
        "--timeline-sequence",
        type=str,
        default="",
        help="Optional sequence name. Empty = use first available sequence.",
    )
    return parser.parse_args()


def locate_pointcloud_file(info: Dict, processed_root: str) -> str:
    pc = info.get("point_cloud", {})
    seq = pc.get("lidar_sequence")
    sample_idx = pc.get("sample_idx")
    if seq is None or sample_idx is None:
        return ""
    return os.path.join(processed_root, seq, f"{int(sample_idx):04d}.npy")


def summarize_points(points: np.ndarray) -> Dict:
    xyz = points[:, :3]
    out = {
        "num_points": int(points.shape[0]),
        "num_features": int(points.shape[1]),
        "x_min": float(np.min(xyz[:, 0])),
        "x_max": float(np.max(xyz[:, 0])),
        "y_min": float(np.min(xyz[:, 1])),
        "y_max": float(np.max(xyz[:, 1])),
        "z_min": float(np.min(xyz[:, 2])),
        "z_max": float(np.max(xyz[:, 2])),
    }
    if points.shape[1] >= 4:
        intensity = points[:, 3]
        out["intensity_mean"] = float(np.mean(intensity))
        out["intensity_p95"] = float(np.percentile(intensity, 95))
    else:
        out["intensity_mean"] = np.nan
        out["intensity_p95"] = np.nan
    return out


def save_bev_plot(points: np.ndarray, path: str, title: str):
    xyz = points[:, :3]
    x = xyz[:, 0]
    y = xyz[:, 1]
    c = xyz[:, 2]  # color by height

    # Subsample for faster plotting on large frames.
    if len(x) > 120_000:
        idx = np.random.choice(len(x), 120_000, replace=False)
        x, y, c = x[idx], y[idx], c[idx]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=0.25, c=c, cmap="viridis", alpha=0.7)
    plt.colorbar(label="z (height)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_timeline_3d_plot(
    infos: List[Dict],
    processed_root: str,
    out_path: str,
    timeline_frames: int,
    timeline_point_cap: int,
    sequence_override: str,
):
    sequence = sequence_override
    if not sequence:
        for info in infos:
            sequence = info.get("point_cloud", {}).get("lidar_sequence", "")
            if sequence:
                break
    if not sequence:
        return False, "No sequence found in infos"

    # Keep only frames from a single sequence so the time progression is meaningful.
    seq_infos = [
        i for i in infos if i.get("point_cloud", {}).get("lidar_sequence", "") == sequence
    ]
    if not seq_infos:
        return False, f"Sequence not found: {sequence}"

    seq_infos = sorted(seq_infos, key=lambda x: int(x.get("point_cloud", {}).get("sample_idx", 0)))
    seq_infos = seq_infos[:timeline_frames]

    xs, ys, zs, ts = [], [], [], []
    loaded = 0
    for t, info in enumerate(seq_infos):
        npy_path = locate_pointcloud_file(info, processed_root)
        if not npy_path or not os.path.exists(npy_path):
            continue
        points = np.load(npy_path)
        xyz = points[:, :3]
        if xyz.shape[0] > 12000:
            idx = np.random.choice(xyz.shape[0], 12000, replace=False)
            xyz = xyz[idx]
        xs.append(xyz[:, 0])
        ys.append(xyz[:, 1])
        zs.append(xyz[:, 2])
        ts.append(np.full(xyz.shape[0], t))
        loaded += 1

    if loaded == 0:
        return False, f"No frame files loaded for sequence: {sequence}"

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    t = np.concatenate(ts)

    if x.shape[0] > timeline_point_cap:
        idx = np.random.choice(x.shape[0], timeline_point_cap, replace=False)
        x, y, z, t = x[idx], y[idx], z[idx], t[idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=t, cmap="plasma", s=0.3, alpha=0.5)
    cb = fig.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label("time step (frame order)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"3D Point Cloud Timeline - {sequence} ({loaded} frames)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

    return True, sequence


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    bev_dir = os.path.join(args.out_dir, "bev_samples")
    os.makedirs(bev_dir, exist_ok=True)

    with open(args.infos, "rb") as f:
        infos = pickle.load(f)

    total = len(infos)
    scan_n = total if args.max_samples == -1 else min(args.max_samples, total)
    subset = infos[:scan_n]

    rows: List[Dict] = []
    missing_files = 0
    plotted = 0

    for i, info in enumerate(tqdm(subset, desc="Reading point clouds")):
        npy_path = locate_pointcloud_file(info, args.processed_root)
        if not npy_path or not os.path.exists(npy_path):
            missing_files += 1
            continue

        points = np.load(npy_path)
        s = summarize_points(points)
        s["frame_idx"] = i
        s["frame_id"] = info.get("frame_id", f"frame_{i}")
        s["npy_path"] = npy_path
        rows.append(s)

        if plotted < args.plot_samples:
            out_png = os.path.join(bev_dir, f"bev_{i:05d}.png")
            save_bev_plot(points, out_png, f"BEV Point Cloud - frame {i}")
            plotted += 1

    if not rows:
        print("No point clouds were loaded. Check --infos and --processed-root paths.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "pointcloud_frame_stats.csv"), index=False)

    # Aggregate plots
    plt.figure(figsize=(8, 5))
    plt.hist(df["num_points"], bins=30)
    plt.title("Point Count per Frame")
    plt.xlabel("num_points")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "num_points_per_frame_hist.png"), dpi=140)
    plt.close()

    if "intensity_mean" in df.columns and df["intensity_mean"].notna().any():
        plt.figure(figsize=(8, 5))
        plt.hist(df["intensity_mean"].dropna(), bins=30)
        plt.title("Mean Intensity per Frame")
        plt.xlabel("mean intensity")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "mean_intensity_per_frame_hist.png"), dpi=140)
        plt.close()

    timeline_path = os.path.join(args.out_dir, "pointcloud_timeline_3d.png")
    ok, timeline_info = save_timeline_3d_plot(
        infos=subset,
        processed_root=args.processed_root,
        out_path=timeline_path,
        timeline_frames=args.timeline_frames,
        timeline_point_cap=args.timeline_point_cap,
        sequence_override=args.timeline_sequence,
    )

    print("\n=== POINT CLOUD EDA SUMMARY ===")
    print(f"Infos file: {args.infos}")
    print(f"Processed root: {args.processed_root}")
    print(f"Total frames in infos: {total}")
    print(f"Frames scanned: {scan_n}")
    print(f"Loaded point-cloud frames: {len(df)}")
    print(f"Missing files: {missing_files}")
    print(f"Avg points/frame: {df['num_points'].mean():.1f}")
    print(f"Median points/frame: {df['num_points'].median():.1f}")
    print(f"Saved to: {args.out_dir}")
    print(" - pointcloud_frame_stats.csv")
    print(" - num_points_per_frame_hist.png")
    if "intensity_mean" in df.columns and df["intensity_mean"].notna().any():
        print(" - mean_intensity_per_frame_hist.png")
    print(" - bev_samples/*.png")
    if ok:
        print(f" - pointcloud_timeline_3d.png (sequence: {timeline_info})")
    else:
        print(f" - timeline plot skipped: {timeline_info}")


if __name__ == "__main__":
    main()
