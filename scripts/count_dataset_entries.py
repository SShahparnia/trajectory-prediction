#!/usr/bin/env python3
"""
Sanity check: count entries in Waymo data on HPC.

Modes:
  (default)     infos pickles + optional processed .npy walk
  --entire       entire --waymo-root: TFRecords, all .npy, all .pkl (one `find` pass)
  --du           with --entire, also run `du -sh` (often VERY slow on NFS; optional timeout)
"""
import argparse
import os
import pickle
import shutil
import subprocess
from pathlib import Path


def count_pkl(path: str) -> int:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return len(data)
    return -1


def count_npy_recursive(root: str) -> int:
    n = 0
    for _dir, _, files in os.walk(root):
        for f in files:
            if f.endswith(".npy"):
                n += 1
    return n


def count_find_stream(root: str, name_pattern: str) -> int:
    """Count files matching find -name (literal). Streams stdout; fast on large trees."""
    try:
        proc = subprocess.Popen(
            ["find", root, "-type", "f", "-name", name_pattern],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        n = 0
        assert proc.stdout is not None
        for _ in proc.stdout:
            n += 1
        proc.wait()
        return n if proc.returncode == 0 else -1
    except OSError:
        return -1


def count_entire_tree_one_find(root: str):
    """
    Single `find -type f` pass; classify by suffix. Much faster than three separate finds on NFS.
    Returns (n_tfrecord, n_npy, n_pkl) or (-1,-1,-1) on failure.
    """
    try:
        proc = subprocess.Popen(
            ["find", root, "-type", "f"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        n_tf = n_npy = n_pkl = 0
        assert proc.stdout is not None
        for line in proc.stdout:
            path = line.decode("utf-8", errors="replace").rstrip("\n")
            low = path.lower()
            if low.endswith(".tfrecord") or low.endswith(".tfrecord.gz"):
                n_tf += 1
            elif low.endswith(".npy"):
                n_npy += 1
            elif low.endswith(".pkl"):
                n_pkl += 1
        proc.wait()
        if proc.returncode != 0:
            return -1, -1, -1
        return n_tf, n_npy, n_pkl
    except OSError:
        return -1, -1, -1


def du_human(root: str, timeout_sec: int = 180) -> str:
    """Total size of root (GNU coreutils du). May block a long time on huge NFS without timeout."""
    du = shutil.which("du")
    if not du:
        return "n/a (du not found)"
    try:
        proc = subprocess.run(
            [du, "-sh", root],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            return "n/a (du failed)"
        return proc.stdout.decode().split()[0]
    except subprocess.TimeoutExpired:
        return f"skipped (du exceeded {timeout_sec}s - NFS may be slow; try again or run du manually)"
    except OSError:
        return "n/a"


def print_entire_tree(root: Path, run_du: bool) -> None:
    r = str(root)
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return

    print("Scanning ENTIRE tree with one `find` pass (may take several minutes on large NFS)...")
    print()

    n_tf, n_npy, n_pkl = count_entire_tree_one_find(r)

    if n_tf < 0:
        print("WARN: `find` failed; falling back to one Python os.walk (slower).")
        n_tf = n_npy = n_pkl = 0
        for _dir, _, files in os.walk(r):
            for f in files:
                low = f.lower()
                if low.endswith(".tfrecord") or low.endswith(".tfrecord.gz"):
                    n_tf += 1
                elif low.endswith(".npy"):
                    n_npy += 1
                elif low.endswith(".pkl"):
                    n_pkl += 1

    print(f"Files *.tfrecord (+ .gz): {n_tf}")
    print(f"Files *.npy:              {n_npy}")
    print(f"Files *.pkl:              {n_pkl}")
    print()
    print("File counts complete (find finished).", flush=True)
    if run_du:
        print(f"Running du -sh (can take minutes on large NFS; timeout 180s)...", flush=True)
        print(f"Disk usage (du -sh): {du_human(r)}", flush=True)
    else:
        print(
            "Skipping disk usage (du). Re-run with --du if you need size; du often stalls on huge NFS.",
            flush=True,
        )
    print()


def main():
    p = argparse.ArgumentParser(description="Count dataset entries (sanity check)")
    p.add_argument(
        "--waymo-root",
        type=str,
        default="/scratch/lts-data/cmpe249-fa22/Waymo132",
        help="Waymo folder on HPC (entire tree when using --entire)",
    )
    p.add_argument(
        "--entire",
        action="store_true",
        help="Scan entire --waymo-root: TFRecords, npy, pkl counts (one find pass)",
    )
    p.add_argument(
        "--du",
        action="store_true",
        help="With --entire, run du -sh on root (often slow; 180s timeout)",
    )
    p.add_argument(
        "--processed-subdir",
        type=str,
        default="waymo_processed_data_v0_5_0",
        help="Subfolder with segment/*/####.npy",
    )
    p.add_argument(
        "--walk-npy",
        action="store_true",
        help="Recursively count all .npy under processed subdir only (not whole tree)",
    )
    args = p.parse_args()

    root = Path(args.waymo_root)
    print(f"Waymo root: {root}")
    print()

    if args.entire:
        print_entire_tree(root, run_du=args.du)
        # still print infos below for apples-to-apples with pipeline
        print("--- Processed infos (frame lists used by our code) ---")
        print()

    # --- infos pickles ---
    for name in (
        "waymo_processed_data_v0_5_0_infos_train.pkl",
        "waymo_processed_data_v0_5_0_infos_val.pkl",
    ):
        path = root / name
        if not path.is_file():
            print(f"MISSING: {path}")
            continue
        n = count_pkl(str(path))
        split = "train" if "train" in name else "val"
        print(f"infos {split}: {n} entries  ({path.name})")

    train_p = root / "waymo_processed_data_v0_5_0_infos_train.pkl"
    val_p = root / "waymo_processed_data_v0_5_0_infos_val.pkl"
    if train_p.is_file() and val_p.is_file():
        nt = count_pkl(str(train_p))
        nv = count_pkl(str(val_p))
        print(f"infos total (train+val): {nt + nv}")

    if args.entire:
        print()
        print(
            "Note: 'infos' rows are the processed frame list your pipeline uses; "
            "TFRecord/*.npy counts include all shards and processed LiDAR under this tree."
        )
        print("DONE.", flush=True)
        return

    print()

    if args.processed_subdir and args.walk_npy:
        proc = root / args.processed_subdir
        if proc.is_dir():
            print(f"Counting .npy files under {proc} ...")
            n_npy = count_npy_recursive(str(proc))
            print(f"total .npy LiDAR frames (recursive): {n_npy}")
        else:
            print(f"Processed dir not found: {proc}")
    elif args.processed_subdir:
        proc = root / args.processed_subdir
        if proc.is_dir():
            n_seg = sum(1 for d in proc.iterdir() if d.is_dir())
            print(f"processed folder: {proc}")
            print(f"segment subfolders (count only, fast): {n_seg}")
            print("(use --walk-npy for .npy under processed only; use --entire for whole HPC folder)")
        else:
            print(f"Processed dir not found: {proc}")

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
