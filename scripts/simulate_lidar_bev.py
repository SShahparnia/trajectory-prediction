"""
Create a simulation-style BEV playback from Waymo processed LiDAR frames.

Outputs:
- PNG frame sequence
- Optional GIF animation

Optional overlays:
- Ground-truth object centers from annos
- Unsupervised cluster centroids (DBSCAN) as object-detection proxy
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.cluster import DBSCAN


def parse_args():
    p = argparse.ArgumentParser("Simulate LiDAR BEV playback")
    p.add_argument(
        "--infos",
        type=str,
        default="data/infos_test.pkl",
        help="Path to infos pickle",
    )
    p.add_argument(
        "--processed-root",
        type=str,
        default="/scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0",
        help="Root directory with sequence folders and frame .npy files",
    )
    p.add_argument(
        "--sequence",
        type=str,
        default="",
        help="Specific lidar_sequence to simulate. Empty uses first available.",
    )
    p.add_argument("--start-idx", type=int, default=0, help="Start frame offset in sequence")
    p.add_argument("--num-frames", type=int, default=30, help="Number of frames to render")
    p.add_argument("--fps", type=int, default=5, help="GIF frames per second")
    p.add_argument(
        "--mode",
        type=str,
        default="bev",
        choices=["bev", "3d"],
        help="Render top-down BEV or full 3D point cloud frames",
    )
    p.add_argument(
        "--point-cap",
        type=int,
        default=70000,
        help="Max points per frame for plotting speed",
    )
    p.add_argument("--overlay-gt", action="store_true", help="Overlay GT object centers")
    p.add_argument(
        "--cluster-detections",
        action="store_true",
        help="Overlay DBSCAN centroids as simple detection proxy",
    )
    p.add_argument("--dbscan-eps", type=float, default=0.9)
    p.add_argument("--dbscan-min-samples", type=int, default=20)
    p.add_argument(
        "--clean",
        action="store_true",
        help="Hide title/axes/grid/colorbar/legend for presentation-clean visuals",
    )
    p.add_argument("--view-elev", type=float, default=26.0, help="3D mode camera elevation")
    p.add_argument("--view-azim", type=float, default=-58.0, help="3D mode camera azimuth")
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/lidar_sim",
        help="Output directory",
    )
    return p.parse_args()


def locate_frame_path(info, processed_root):
    pc = info.get("point_cloud", {})
    seq = pc.get("lidar_sequence", "")
    sample_idx = pc.get("sample_idx", None)
    if not seq or sample_idx is None:
        return ""
    return os.path.join(processed_root, seq, "{:04d}.npy".format(int(sample_idx)))


def gt_centers(info):
    ann = info.get("annos", {})
    loc = np.asarray(ann.get("location", np.zeros((0, 3), dtype=np.float32)))
    if loc.ndim != 2 or loc.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return loc[:, :2].astype(np.float32)


def dbscan_centroids(points_xy, eps, min_samples):
    if points_xy.shape[0] < min_samples:
        return np.zeros((0, 2), dtype=np.float32)
    cl = DBSCAN(eps=eps, min_samples=min_samples)
    labels = cl.fit_predict(points_xy)
    valid = labels >= 0
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32)
    out = []
    for lb in np.unique(labels[valid]):
        pts = points_xy[labels == lb]
        out.append(np.mean(pts, axis=0))
    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(out, axis=0).astype(np.float32)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    with open(args.infos, "rb") as f:
        infos = pickle.load(f)
    if not infos:
        raise RuntimeError("infos file is empty")

    seq = args.sequence
    if not seq:
        for i in infos:
            seq = i.get("point_cloud", {}).get("lidar_sequence", "")
            if seq:
                break
    if not seq:
        raise RuntimeError("No sequence found in infos")

    seq_infos = [i for i in infos if i.get("point_cloud", {}).get("lidar_sequence", "") == seq]
    seq_infos = sorted(seq_infos, key=lambda x: int(x.get("point_cloud", {}).get("sample_idx", 0)))
    if not seq_infos:
        raise RuntimeError("Sequence not found in infos: {}".format(seq))

    start = max(0, args.start_idx)
    end = min(len(seq_infos), start + max(1, args.num_frames))
    seq_infos = seq_infos[start:end]
    if not seq_infos:
        raise RuntimeError("No frames selected after start/num filters")

    rendered = []
    for t, info in enumerate(seq_infos):
        npy_path = locate_frame_path(info, args.processed_root)
        if not npy_path or not os.path.exists(npy_path):
            continue
        points = np.load(npy_path)
        xyz = points[:, :3]
        if xyz.shape[0] > args.point_cap:
            idx = np.random.choice(xyz.shape[0], args.point_cap, replace=False)
            xyz = xyz[idx]

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        fig = plt.figure(figsize=(7, 7))
        if args.mode == "3d":
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x, y, z, s=0.2, c=z, cmap="viridis", alpha=0.6)
            ax.view_init(elev=args.view_elev, azim=args.view_azim)
        else:
            ax = fig.add_subplot(111)
            ax.scatter(x, y, s=0.2, c=z, cmap="viridis", alpha=0.65)

        if args.overlay_gt:
            centers = gt_centers(info)
            if centers.shape[0] > 0:
                if args.mode == "3d":
                    ax.scatter(
                        centers[:, 0],
                        centers[:, 1],
                        np.zeros_like(centers[:, 0]),
                        s=20,
                        marker="o",
                        facecolors="none",
                        edgecolors="lime",
                        linewidths=0.8,
                        label="GT objects",
                    )
                else:
                    ax.scatter(
                        centers[:, 0],
                        centers[:, 1],
                        s=20,
                        marker="o",
                        facecolors="none",
                        edgecolors="lime",
                        linewidths=0.8,
                        label="GT objects",
                    )

        if args.cluster_detections:
            near = xyz[np.abs(z) < 2.5][:, :2]
            ctrs = dbscan_centroids(near, args.dbscan_eps, args.dbscan_min_samples)
            if ctrs.shape[0] > 0:
                if args.mode == "3d":
                    ax.scatter(
                        ctrs[:, 0],
                        ctrs[:, 1],
                        np.zeros_like(ctrs[:, 0]),
                        s=24,
                        marker="x",
                        c="red",
                        linewidths=0.9,
                        label="Cluster detections",
                    )
                else:
                    ax.scatter(
                        ctrs[:, 0],
                        ctrs[:, 1],
                        s=24,
                        marker="x",
                        c="red",
                        linewidths=0.9,
                        label="Cluster detections",
                    )

        pc = info.get("point_cloud", {})
        sid = int(pc.get("sample_idx", t))
        if not args.clean:
            if args.mode == "3d":
                ax.set_title("LiDAR 3D Simulation | {} | frame {:04d}".format(seq, sid))
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_zlabel("z (m)")
            else:
                ax.set_title("LiDAR BEV Simulation | {} | frame {:04d}".format(seq, sid))
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.axis("equal")
                ax.grid(True, alpha=0.2)
            if args.overlay_gt or args.cluster_detections:
                ax.legend(loc="upper right", fontsize=7)
        else:
            ax.set_axis_off()

        plt.tight_layout()
        out_png = os.path.join(frames_dir, "frame_{:03d}.png".format(t))
        plt.savefig(out_png, dpi=140)
        plt.close(fig)
        rendered.append(out_png)

    if not rendered:
        raise RuntimeError("No frames rendered; check infos/processed-root paths")

    # Build GIF from saved PNGs.
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.axis("off")
    im = ax.imshow(plt.imread(rendered[0]))

    def update(i):
        im.set_array(plt.imread(rendered[i]))
        return [im]

    anim = FuncAnimation(fig, update, frames=len(rendered), interval=int(1000 / max(1, args.fps)))
    gif_path = os.path.join(args.out_dir, "lidar_bev_sim.gif")
    anim.save(gif_path, writer=PillowWriter(fps=max(1, args.fps)))
    plt.close(fig)

    print("Rendered frames: {}".format(len(rendered)))
    print("Frames dir: {}".format(frames_dir))
    print("GIF: {}".format(gif_path))


if __name__ == "__main__":
    main()
