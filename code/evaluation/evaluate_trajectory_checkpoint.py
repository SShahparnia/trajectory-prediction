import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from code.data_pipeline.multi_agent_windows import (  # noqa: E402
    MultiAgentTrajectoryDataset,
    build_multi_agent_windows,
)
from code.data_pipeline.waymo_windows import TrajectoryDataset, build_xy_windows  # noqa: E402
from code.evaluation.metrics import ade, fde  # noqa: E402
from code.models.lstm_baseline import LSTMBaseline  # noqa: E402
from code.models.multi_agent_lstm import MultiAgentLSTM  # noqa: E402
from code.models.transformer_baseline import TransformerBaseline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("Evaluate trajectory checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--infos", type=str, default="data/infos_test.pkl")
    p.add_argument("--past-len", type=int, default=10)
    p.add_argument("--future-len", type=int, default=20)
    p.add_argument("--max-windows", type=int, default=4000)
    p.add_argument("--max-neighbors", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--metrics-out", type=str, required=True)
    p.add_argument(
        "--viz-out-dir",
        type=str,
        default="",
        help="Optional directory to save qualitative trajectory plots.",
    )
    p.add_argument("--num-viz", type=int, default=12)
    return p.parse_args()


def _build_model(model_type, ckpt_args, future_len):
    if model_type == "lstm":
        return LSTMBaseline(
            in_dim=2,
            hidden_dim=ckpt_args.get("hidden_dim", 128),
            future_len=future_len,
        )
    if model_type == "transformer":
        return TransformerBaseline(
            in_dim=2,
            d_model=ckpt_args.get("hidden_dim", 128),
            nhead=ckpt_args.get("transformer_heads", 4),
            num_layers=ckpt_args.get("transformer_layers", 2),
            dim_feedforward=ckpt_args.get("transformer_ffn_dim", 256),
            dropout=ckpt_args.get("transformer_dropout", 0.1),
            future_len=future_len,
        )
    return MultiAgentLSTM(
        in_dim=2,
        hidden_dim=ckpt_args.get("hidden_dim", 128),
        neighbor_hidden_dim=max(ckpt_args.get("hidden_dim", 128) // 2, 32),
        future_len=future_len,
    )


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_type = ckpt.get("model_type", "lstm")
    ckpt_args = ckpt.get("args", {})

    model = _build_model(model_type, ckpt_args, args.future_len)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds, tgts = [], []
    past_inputs = []
    neigh_inputs = []
    neigh_masks = []
    if model_type == "multi_lstm":
        ds = MultiAgentTrajectoryDataset(
            *build_multi_agent_windows(
                infos_path=args.infos,
                past_len=args.past_len,
                future_len=args.future_len,
                max_neighbors=args.max_neighbors,
                max_windows=args.max_windows,
            )
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for ego_x, neigh_x, neigh_mask, yb in loader:
                pred = model(ego_x, neigh_x, neigh_mask)
                preds.append(pred.numpy())
                tgts.append(yb.numpy())
                past_inputs.append(ego_x.numpy())
                neigh_inputs.append(neigh_x.numpy())
                neigh_masks.append(neigh_mask.numpy())
    else:
        ds = TrajectoryDataset(
            *build_xy_windows(
                infos_path=args.infos,
                past_len=args.past_len,
                future_len=args.future_len,
                max_windows=args.max_windows,
            )
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for xb, yb in loader:
                pred = model(xb)
                preds.append(pred.numpy())
                tgts.append(yb.numpy())
                past_inputs.append(xb.numpy())

    pred = np.concatenate(preds, axis=0)
    tgt = np.concatenate(tgts, axis=0)
    out = {
        "model_type": model_type,
        "checkpoint": args.checkpoint,
        "infos": args.infos,
        "past_len": args.past_len,
        "future_len": args.future_len,
        "max_windows": args.max_windows,
        "samples": len(ds),
        "ADE": float(ade(pred, tgt)),
        "FDE": float(fde(pred, tgt)),
    }
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"saved metrics: {args.metrics_out}")

    if args.viz_out_dir:
        os.makedirs(args.viz_out_dir, exist_ok=True)
        n = min(args.num_viz, pred.shape[0])
        past_np = np.concatenate(past_inputs, axis=0)
        if model_type == "multi_lstm":
            neigh_np = np.concatenate(neigh_inputs, axis=0)
            mask_np = np.concatenate(neigh_masks, axis=0)
        for i in range(n):
            fig = plt.figure(figsize=(6, 6))
            if model_type == "multi_lstm":
                for k in range(neigh_np.shape[1]):
                    if mask_np[i, k] > 0.5:
                        ntraj = neigh_np[i, k]
                        plt.plot(
                            ntraj[:, 0],
                            ntraj[:, 1],
                            color="lightgray",
                            linewidth=1.0,
                            alpha=0.8,
                        )
            plt.plot(
                past_np[i, :, 0],
                past_np[i, :, 1],
                "o-",
                color="black",
                label="past",
                linewidth=1.5,
                markersize=3,
            )
            plt.plot(
                tgt[i, :, 0],
                tgt[i, :, 1],
                "o-",
                color="tab:green",
                label="future_gt",
                linewidth=1.8,
                markersize=3,
            )
            plt.plot(
                pred[i, :, 0],
                pred[i, :, 1],
                "o--",
                color="tab:red",
                label="future_pred",
                linewidth=1.8,
                markersize=3,
            )
            plt.scatter([0.0], [0.0], color="tab:blue", s=35, label="ego_anchor")
            plt.axis("equal")
            plt.grid(True, alpha=0.3)
            plt.title("{} sample {}".format(model_type, i))
            plt.xlabel("x_local (m)")
            plt.ylabel("y_local (m)")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            out_png = os.path.join(args.viz_out_dir, "traj_{:03d}.png".format(i))
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
        print("saved qualitative plots: {}".format(args.viz_out_dir))


if __name__ == "__main__":
    main()
