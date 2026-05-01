import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
from code.evaluation.metrics import ade, fde, min_ade_at_k, min_fde_at_k  # noqa: E402
from code.models.lstm_baseline import LSTMBaseline  # noqa: E402
from code.models.multi_agent_lstm import MultiAgentLSTM  # noqa: E402
from code.models.multi_agent_lstm_attn import MultiAgentLSTMAttention  # noqa: E402
from code.models.multimodal_lstm import MultimodalLSTM  # noqa: E402
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
    if model_type == "multi_lstm":
        return MultiAgentLSTM(
            in_dim=2,
            hidden_dim=ckpt_args.get("hidden_dim", 128),
            neighbor_hidden_dim=max(ckpt_args.get("hidden_dim", 128) // 2, 32),
            future_len=future_len,
        )
    if model_type == "multi_lstm_attn":
        return MultiAgentLSTMAttention(
            in_dim=2,
            hidden_dim=ckpt_args.get("hidden_dim", 128),
            neighbor_hidden_dim=max(ckpt_args.get("hidden_dim", 128) // 2, 32),
            future_len=future_len,
            attn_dim=ckpt_args.get("hidden_dim", 128),
            n_heads=ckpt_args.get("attn_heads", 4),
            dropout=ckpt_args.get("transformer_dropout", 0.1),
        )
    if model_type == "lstm_multimodal":
        return MultimodalLSTM(
            in_dim=2,
            hidden_dim=ckpt_args.get("hidden_dim", 128),
            future_len=future_len,
            num_modes=ckpt_args.get("num_modes", 6),
        )
    raise ValueError("unknown model_type in checkpoint: {}".format(model_type))


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
    if model_type in ("multi_lstm", "multi_lstm_attn"):
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
    if model_type == "lstm_multimodal":
        ade_v, fde_v = float(min_ade_at_k(pred, tgt)), float(min_fde_at_k(pred, tgt))
    else:
        ade_v, fde_v = float(ade(pred, tgt)), float(fde(pred, tgt))
    out = {
        "model_type": model_type,
        "checkpoint": args.checkpoint,
        "infos": args.infos,
        "past_len": args.past_len,
        "future_len": args.future_len,
        "max_windows": args.max_windows,
        "samples": len(ds),
        "ADE": ade_v,
        "FDE": fde_v,
    }
    if model_type == "lstm_multimodal":
        out["minADE_at_K"] = ade_v
        out["minFDE_at_K"] = fde_v
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"saved metrics: {args.metrics_out}")

    if args.viz_out_dir:
        os.makedirs(args.viz_out_dir, exist_ok=True)
        n = min(args.num_viz, pred.shape[0])
        past_np = np.concatenate(past_inputs, axis=0)
        if model_type in ("multi_lstm", "multi_lstm_attn"):
            neigh_np = np.concatenate(neigh_inputs, axis=0)
            mask_np = np.concatenate(neigh_masks, axis=0)
        sns.set_theme(
            style="whitegrid",
            context="notebook",
            font_scale=0.85,
            rc={"grid.alpha": 0.35, "grid.linestyle": "--"},
        )
        c_neigh, c_past, c_gt, c_pred, c_ego = sns.color_palette("muted", 5)
        for i in range(n):
            fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=150)
            if model_type in ("multi_lstm", "multi_lstm_attn"):
                for k in range(neigh_np.shape[1]):
                    if mask_np[i, k] > 0.5:
                        ntraj = neigh_np[i, k]
                        ax.plot(
                            ntraj[:, 0],
                            ntraj[:, 1],
                            color=c_neigh,
                            linewidth=1.0,
                            alpha=0.75,
                            solid_capstyle="round",
                        )
            ax.plot(
                past_np[i, :, 0],
                past_np[i, :, 1],
                "o-",
                color=c_past,
                label="past",
                linewidth=2.0,
                markersize=4,
                solid_capstyle="round",
            )
            ax.plot(
                tgt[i, :, 0],
                tgt[i, :, 1],
                "o-",
                color=c_gt,
                label="future (GT)",
                linewidth=2.2,
                markersize=4,
                solid_capstyle="round",
            )
            if model_type == "lstm_multimodal":
                modes = pred[i]  # [K, F, 2]
                err_k = np.linalg.norm(modes - tgt[i][np.newaxis, :, :], axis=-1).mean(axis=1)
                best = int(np.argmin(err_k))
                for kk in range(modes.shape[0]):
                    if kk != best:
                        ax.plot(
                            modes[kk, :, 0],
                            modes[kk, :, 1],
                            "--",
                            color="#b2bec3",
                            linewidth=0.9,
                            alpha=0.55,
                        )
                ax.plot(
                    modes[best, :, 0],
                    modes[best, :, 1],
                    "o--",
                    color=c_pred,
                    label="future (best mode)",
                    linewidth=2.2,
                    markersize=4,
                    solid_capstyle="round",
                )
            else:
                ax.plot(
                    pred[i, :, 0],
                    pred[i, :, 1],
                    "o--",
                    color=c_pred,
                    label="future (pred)",
                    linewidth=2.2,
                    markersize=4,
                    solid_capstyle="round",
                )
            ax.scatter([0.0], [0.0], color=c_ego, s=42, zorder=5, edgecolors="white", linewidths=0.8, label="ego anchor")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("{} — sample {}".format(model_type, i))
            ax.set_xlabel("x_local (m)")
            ax.set_ylabel("y_local (m)")
            ax.legend(loc="best", fontsize=8, framealpha=0.95)
            sns.despine(ax=ax, left=False, bottom=False)
            plt.tight_layout()
            out_png = os.path.join(args.viz_out_dir, "traj_{:03d}.png".format(i))
            fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        print("saved qualitative plots: {}".format(args.viz_out_dir))


if __name__ == "__main__":
    main()
