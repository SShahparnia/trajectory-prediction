import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from traj_code.data_pipeline.waymo_windows import TrajectoryDataset, build_xy_windows
from traj_code.evaluation.metrics import ade, fde
from traj_code.models.lstm_baseline import LSTMBaseline


def parse_args():
    p = argparse.ArgumentParser("Train LSTM baseline on Waymo windows")
    p.add_argument(
        "--infos",
        type=str,
        default="/scratch/lts-data/cmpe249-fa22/Waymo132/waymo_processed_data_v0_5_0_infos_train.pkl",
    )
    p.add_argument("--past-len", type=int, default=10)
    p.add_argument("--future-len", type=int, default=20)
    p.add_argument("--max-windows", type=int, default=12000)
    p.add_argument(
        "--val-infos",
        type=str,
        default="",
        help="Optional explicit validation infos pickle. If empty, uses random 80/20 split.",
    )
    p.add_argument(
        "--val-max-windows",
        type=int,
        default=6000,
        help="Max validation windows when --val-infos is set.",
    )
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="results/train_lstm_baseline")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y = build_xy_windows(
        infos_path=args.infos,
        past_len=args.past_len,
        future_len=args.future_len,
        max_windows=args.max_windows,
    )
    train_ds = TrajectoryDataset(x, y)

    if args.val_infos:
        x_val, y_val = build_xy_windows(
            infos_path=args.val_infos,
            past_len=args.past_len,
            future_len=args.future_len,
            max_windows=args.val_max_windows,
        )
        val_ds = TrajectoryDataset(x_val, y_val)
        split_mode = "explicit_val_infos"
    else:
        n_train = int(0.8 * len(train_ds))
        n_val = len(train_ds) - n_train
        train_ds, val_ds = random_split(
            train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )
        split_mode = "random_80_20"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMBaseline(in_dim=2, hidden_dim=args.hidden_dim, future_len=args.future_len).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.SmoothL1Loss()

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        val_pred, val_tgt = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                va_losses.append(loss_fn(pred, yb).item())
                val_pred.append(pred.cpu().numpy())
                val_tgt.append(yb.cpu().numpy())

        val_pred = np.concatenate(val_pred, axis=0)
        val_tgt = np.concatenate(val_tgt, axis=0)
        val_ade = ade(val_pred, val_tgt)
        val_fde = fde(val_pred, val_tgt)
        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))

        row = {
            "epoch": epoch,
            "train_loss": tr,
            "val_loss": va,
            "val_ADE": val_ade,
            "val_FDE": val_fde,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={tr:.4f} val_loss={va:.4f} "
            f"val_ADE={val_ade:.4f} val_FDE={val_fde:.4f}"
        )

        if va < best_val:
            best_val = va
            ckpt = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_loss": va,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best_model.pt"))

    # save history csv
    history_path = os.path.join(args.out_dir, "train_history.csv")
    with open(history_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_ADE,val_FDE\n")
        for r in history:
            f.write(
                f"{r['epoch']},{r['train_loss']:.6f},{r['val_loss']:.6f},"
                f"{r['val_ADE']:.6f},{r['val_FDE']:.6f}\n"
            )

    print(f"saved checkpoint: {os.path.join(args.out_dir, 'best_model.pt')}")
    print(f"saved history: {history_path}")
    print(f"train windows: {len(train_ds)}")
    print(f"val windows: {len(val_ds)}")
    print(f"split mode: {split_mode}")

    run_info = {
        "args": vars(args),
        "split_mode": split_mode,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "best_val_loss": float(best_val),
    }
    run_info_path = os.path.join(args.out_dir, "run_info.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"saved run info: {run_info_path}")


if __name__ == "__main__":
    main()
