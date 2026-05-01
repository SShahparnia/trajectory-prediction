import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from code.data_pipeline.multi_agent_windows import (  # noqa: E402
    MultiAgentTrajectoryDataset,
    build_multi_agent_windows,
)
from code.data_pipeline.waymo_windows import TrajectoryDataset, build_xy_windows  # noqa: E402
from code.evaluation.metrics import ade, fde, min_ade_at_k, min_fde_at_k  # noqa: E402
from code.models.lstm_baseline import LSTMBaseline  # noqa: E402
from code.models.multi_agent_lstm import MultiAgentLSTM  # noqa: E402
from code.models.multi_agent_lstm_attn import MultiAgentLSTMAttention  # noqa: E402
from code.models.multimodal_lstm import MultimodalLSTM, multimodal_wta_smooth_l1  # noqa: E402
from code.models.transformer_baseline import TransformerBaseline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("Train trajectory model variants")
    p.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["lstm", "transformer", "multi_lstm", "multi_lstm_attn", "lstm_multimodal"],
    )
    p.add_argument("--train-infos", type=str, default="data/infos_train.pkl")
    p.add_argument("--val-infos", type=str, default="data/infos_val.pkl")
    p.add_argument("--past-len", type=int, default=10)
    p.add_argument("--future-len", type=int, default=20)
    p.add_argument("--max-train-windows", type=int, default=12000)
    p.add_argument("--max-val-windows", type=int, default=4000)
    p.add_argument("--max-neighbors", type=int, default=12)

    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--transformer-heads", type=int, default=4)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--transformer-ffn-dim", type=int, default=256)
    p.add_argument("--transformer-dropout", type=float, default=0.1)

    p.add_argument("--num-modes", type=int, default=6, help="For lstm_multimodal: number of trajectory hypotheses.")
    p.add_argument(
        "--attn-heads",
        type=int,
        default=4,
        help="For multi_lstm_attn: cross-attention heads.",
    )

    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def _build_single_agent_ds(args):
    x_tr, y_tr = build_xy_windows(
        infos_path=args.train_infos,
        past_len=args.past_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
    )
    x_va, y_va = build_xy_windows(
        infos_path=args.val_infos,
        past_len=args.past_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
    )
    return TrajectoryDataset(x_tr, y_tr), TrajectoryDataset(x_va, y_va)


def _build_multi_agent_ds(args):
    tr = build_multi_agent_windows(
        infos_path=args.train_infos,
        past_len=args.past_len,
        future_len=args.future_len,
        max_neighbors=args.max_neighbors,
        max_windows=args.max_train_windows,
    )
    va = build_multi_agent_windows(
        infos_path=args.val_infos,
        past_len=args.past_len,
        future_len=args.future_len,
        max_neighbors=args.max_neighbors,
        max_windows=args.max_val_windows,
    )
    return (
        MultiAgentTrajectoryDataset(*tr),
        MultiAgentTrajectoryDataset(*va),
    )


def _build_model(args):
    if args.model_type == "lstm":
        return LSTMBaseline(in_dim=2, hidden_dim=args.hidden_dim, future_len=args.future_len)
    if args.model_type == "transformer":
        return TransformerBaseline(
            in_dim=2,
            d_model=args.hidden_dim,
            nhead=args.transformer_heads,
            num_layers=args.transformer_layers,
            dim_feedforward=args.transformer_ffn_dim,
            dropout=args.transformer_dropout,
            future_len=args.future_len,
        )
    if args.model_type == "multi_lstm":
        return MultiAgentLSTM(
            in_dim=2,
            hidden_dim=args.hidden_dim,
            neighbor_hidden_dim=max(args.hidden_dim // 2, 32),
            future_len=args.future_len,
        )
    if args.model_type == "multi_lstm_attn":
        return MultiAgentLSTMAttention(
            in_dim=2,
            hidden_dim=args.hidden_dim,
            neighbor_hidden_dim=max(args.hidden_dim // 2, 32),
            future_len=args.future_len,
            attn_dim=args.hidden_dim,
            n_heads=args.attn_heads,
            dropout=args.transformer_dropout,
        )
    if args.model_type == "lstm_multimodal":
        return MultimodalLSTM(
            in_dim=2,
            hidden_dim=args.hidden_dim,
            future_len=args.future_len,
            num_modes=args.num_modes,
        )
    raise ValueError("unknown model_type: {}".format(args.model_type))


def _forward_batch(model_type, model, batch, device):
    if model_type in ("multi_lstm", "multi_lstm_attn"):
        ego_x, neigh_x, neigh_mask, yb = batch
        pred = model(ego_x.to(device), neigh_x.to(device), neigh_mask.to(device))
        return pred, yb.to(device)
    xb, yb = batch
    pred = model(xb.to(device))
    return pred, yb.to(device)


def _batch_loss(loss_fn, model_type, pred, yb):
    if model_type == "lstm_multimodal":
        return multimodal_wta_smooth_l1(pred, yb)
    return loss_fn(pred, yb)


def _val_metrics_numpy(model_type, pred_np, tgt_np):
    if model_type == "lstm_multimodal":
        return float(min_ade_at_k(pred_np, tgt_np)), float(min_fde_at_k(pred_np, tgt_np))
    return float(ade(pred_np, tgt_np)), float(fde(pred_np, tgt_np))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_type in ("multi_lstm", "multi_lstm_attn"):
        train_ds, val_ds = _build_multi_agent_ds(args)
    else:
        train_ds, val_ds = _build_single_agent_ds(args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.SmoothL1Loss()

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for batch in train_loader:
            opt.zero_grad()
            pred, yb = _forward_batch(args.model_type, model, batch, device)
            loss = _batch_loss(loss_fn, args.model_type, pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        pred_all, tgt_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred, yb = _forward_batch(args.model_type, model, batch, device)
                va_losses.append(_batch_loss(loss_fn, args.model_type, pred, yb).item())
                pred_all.append(pred.cpu().numpy())
                tgt_all.append(yb.cpu().numpy())

        pred_np = np.concatenate(pred_all, axis=0)
        tgt_np = np.concatenate(tgt_all, axis=0)
        v_ade, v_fde = _val_metrics_numpy(args.model_type, pred_np, tgt_np)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(tr_losses)),
            "val_loss": float(np.mean(va_losses)),
            "val_ADE": v_ade,
            "val_FDE": v_fde,
        }
        history.append(row)
        print(
            f"[{args.model_type}] epoch={epoch} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_ADE={row['val_ADE']:.4f} val_FDE={row['val_FDE']:.4f}"
        )

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            ckpt = {
                "model_type": args.model_type,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_epoch": epoch,
                "best_val_loss": best_val,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best_model.pt"))

    hist_path = os.path.join(args.out_dir, "train_history.csv")
    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_ADE", "val_FDE"],
        )
        writer.writeheader()
        writer.writerows(history)

    run_info = {
        "model_type": args.model_type,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "best_val_loss": float(best_val),
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"saved checkpoint: {os.path.join(args.out_dir, 'best_model.pt')}")
    print(f"saved history: {hist_path}")
    print(f"saved run info: {os.path.join(args.out_dir, 'run_info.json')}")


if __name__ == "__main__":
    main()
