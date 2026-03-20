import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from code.data_pipeline.waymo_windows import TrajectoryDataset, build_xy_windows
from code.evaluation.metrics import ade, fde
from code.models.lstm_baseline import LSTMBaseline


def parse_args():
    p = argparse.ArgumentParser("Evaluate saved trajectory checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--infos",
        type=str,
        default="/scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0_infos_train.pkl",
    )
    p.add_argument("--past-len", type=int, default=10)
    p.add_argument("--future-len", type=int, default=20)
    p.add_argument("--max-windows", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    x, y = build_xy_windows(
        infos_path=args.infos,
        past_len=args.past_len,
        future_len=args.future_len,
        max_windows=args.max_windows,
    )
    ds = TrajectoryDataset(x, y)
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    _, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    hidden_dim = ckpt.get("args", {}).get("hidden_dim", 128)
    model = LSTMBaseline(in_dim=2, hidden_dim=hidden_dim, future_len=args.future_len)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            preds.append(pred.numpy())
            tgts.append(yb.numpy())

    pred = np.concatenate(preds, axis=0)
    tgt = np.concatenate(tgts, axis=0)
    print(f"ADE: {ade(pred, tgt):.6f}")
    print(f"FDE: {fde(pred, tgt):.6f}")


if __name__ == "__main__":
    main()
