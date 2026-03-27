#!/usr/bin/env python3
"""
Build expanded train/val/test metadata under data/ from ALL frames in the
processed Waymo infos pickles (train + val combined), split by SEQUENCE so the
same segment does not appear in two splits.

Does not copy LiDAR .npy files (huge); infos dicts still point to original paths.

Output (default):
  data/infos_train.pkl
  data/infos_val.pkl
  data/infos_test.pkl
  data/manifest.json
  data/README.txt
"""
import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_infos(path: str) -> List[Dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def sequence_of(info: Dict) -> str:
    return info.get("point_cloud", {}).get("lidar_sequence", "") or ""


def build_splits(
    all_infos: List[Dict],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split infos by unique lidar_sequence; stratify frames into train/val/test."""
    by_seq: Dict[str, List[Dict]] = defaultdict(list)
    for info in all_infos:
        seq = sequence_of(info)
        if not seq:
            continue
        by_seq[seq].append(info)

    seqs = sorted(by_seq.keys())
    rng = random.Random(seed)
    rng.shuffle(seqs)

    n = len(seqs)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = max(0, n - n_train - n_val)

    train_seqs = set(seqs[:n_train])
    val_seqs = set(seqs[n_train : n_train + n_val])
    test_seqs = set(seqs[n_train + n_val :])

    def flatten(seq_set):
        out = []
        for s in sorted(seq_set):
            rows = by_seq[s]
            rows.sort(key=lambda x: int(x.get("point_cloud", {}).get("sample_idx", 0)))
            out.extend(rows)
        return out

    return flatten(train_seqs), flatten(val_seqs), flatten(test_seqs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--waymo-root",
        type=str,
        default="/scratch/lts-data/cmpe249-fa22/Waymo132",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output directory (created if missing)",
    )
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    # test = remainder
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.waymo_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = root / "waymo_processed_data_v0_5_0_infos_train.pkl"
    val_path = root / "waymo_processed_data_v0_5_0_infos_val.pkl"

    for path in (train_path, val_path):
        if not path.is_file():
            raise SystemExit(f"Missing input: {path}")

    train_infos = load_infos(str(train_path))
    val_infos = load_infos(str(val_path))
    combined = train_infos + val_infos

    tr, va, te = build_splits(
        combined,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    def save(name: str, rows: List[Dict]):
        dst = out / name
        with open(dst, "wb") as f:
            pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
        return dst

    f_train = save("infos_train.pkl", tr)
    f_val = save("infos_val.pkl", va)
    f_test = save("infos_test.pkl", te)

    manifest = {
        "source_train_pkl": str(train_path),
        "source_val_pkl": str(val_path),
        "combined_total_frames": len(combined),
        "split_train_frames": len(tr),
        "split_val_frames": len(va),
        "split_test_frames": len(te),
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": 1.0 - args.train_frac - args.val_frac,
        "seed": args.seed,
        "split_by": "lidar_sequence (no segment shared across splits)",
        "outputs": {
            "train": str(f_train),
            "val": str(f_val),
            "test": str(f_test),
        },
    }
    man_path = out / "manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    readme = out / "README.txt"
    with open(readme, "w") as f:
        f.write(
            "Expanded Waymo processed infos splits\n"
            "=====================================\n\n"
            "Built from ALL entries in:\n"
            f"  - {train_path.name}\n"
            f"  - {val_path.name}\n\n"
            f"Total frames combined: {len(combined)}\n"
            f"  train: {len(tr)} frames\n"
            f"  val:   {len(va)} frames\n"
            f"  test:  {len(te)} frames\n\n"
            "Split is by lidar_sequence (whole segments) to avoid leakage.\n"
            "Point cloud paths inside each info still reference /scratch/... originals.\n\n"
            "Point training code at these pickles when you are ready (not wired yet).\n"
        )

    print(json.dumps(manifest, indent=2))
    print(f"\nWrote: {f_train}, {f_val}, {f_test}")
    print(f"Manifest: {man_path}")
    print(f"README: {readme}")


if __name__ == "__main__":
    main()
