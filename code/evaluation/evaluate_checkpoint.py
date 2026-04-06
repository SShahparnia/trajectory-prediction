import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

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
    p.add_argument(
        "--metrics-out",
        type=str,
        default="",
        help="Optional path to save ADE/FDE JSON output.",
    )
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
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

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
    ade_v = float(ade(pred, tgt))
    fde_v = float(fde(pred, tgt))
    print(f"ADE: {ade_v:.6f}")
    print(f"FDE: {fde_v:.6f}")
    print(f"samples: {len(ds)}")

    if args.metrics_out:
        out = {
            "checkpoint": args.checkpoint,
            "infos": args.infos,
            "past_len": args.past_len,
            "future_len": args.future_len,
            "max_windows": args.max_windows,
            "samples": len(ds),
            "ADE": ade_v,
            "FDE": fde_v,
        }
        out_dir = os.path.dirname(args.metrics_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
