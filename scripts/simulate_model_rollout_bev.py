#!/usr/bin/env python3
"""
Simulate a model rollout on a random (or chosen) trajectory window sample.

This visual is model-based (not just raw LiDAR playback):
- Past trajectory (input)
- Ground-truth future
- Predicted future from checkpoint
- Optional neighbor trajectories for multi-agent model
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from code.data_pipeline.multi_agent_windows import build_multi_agent_windows  # noqa: E402
from code.data_pipeline.waymo_windows import build_xy_windows  # noqa: E402
from code.models.lstm_baseline import LSTMBaseline  # noqa: E402
from code.models.multi_agent_lstm import MultiAgentLSTM  # noqa: E402
from code.models.transformer_baseline import TransformerBaseline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("Simulate model trajectory rollout")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--infos", type=str, default="data/infos_test.pkl")
    p.add_argument("--past-len", type=int, default=10)
    p.add_argument("--future-len", type=int, default=20)
    p.add_argument("--max-windows", type=int, default=1200)
    p.add_argument("--max-neighbors", type=int, default=12)
    p.add_argument(
        "--sample-idx",
        type=int,
        default=-1,
        help="Sample index to visualize; -1 selects random sample.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--clean", action="store_true", help="Hide labels/legend for clean slide visuals.")
    p.add_argument("--out-dir", type=str, default="results/model_rollout_sim")
    return p.parse_args()


def build_model(model_type, ckpt_args, future_len):
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_type = ckpt.get("model_type", "lstm")
    ckpt_args = ckpt.get("args", {})
    model = build_model(model_type, ckpt_args, args.future_len)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    neigh = None
    neigh_mask = None
    if model_type == "multi_lstm":
        ego_x, neigh_x, nmask, y = build_multi_agent_windows(
            infos_path=args.infos,
            past_len=args.past_len,
            future_len=args.future_len,
            max_neighbors=args.max_neighbors,
            max_windows=args.max_windows,
        )
    else:
        ego_x, y = build_xy_windows(
            infos_path=args.infos,
            past_len=args.past_len,
            future_len=args.future_len,
            max_windows=args.max_windows,
        )
        neigh_x = None
        nmask = None

    n = ego_x.shape[0]
    if n == 0:
        raise RuntimeError("No samples available for rollout simulation.")
    if args.sample_idx >= 0:
        si = min(args.sample_idx, n - 1)
    else:
        si = random.randrange(n)

    ego = ego_x[si]
    tgt = y[si]
    if model_type == "multi_lstm":
        neigh = neigh_x[si]
        neigh_mask = nmask[si]

    with torch.no_grad():
        ego_t = torch.from_numpy(ego).unsqueeze(0)
        if model_type == "multi_lstm":
            neigh_t = torch.from_numpy(neigh).unsqueeze(0)
            mask_t = torch.from_numpy(neigh_mask).unsqueeze(0)
            pred = model(ego_t, neigh_t, mask_t)[0].numpy()
        else:
            pred = model(ego_t)[0].numpy()

    total_frames = args.past_len + args.future_len
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    x_all = np.concatenate([ego[:, 0], tgt[:, 0], pred[:, 0]])
    y_all = np.concatenate([ego[:, 1], tgt[:, 1], pred[:, 1]])
    pad = 2.0
    xlim = (float(np.min(x_all) - pad), float(np.max(x_all) + pad))
    ylim = (float(np.min(y_all) - pad), float(np.max(y_all) + pad))

    def draw(frame):
        ax.clear()
        if model_type == "multi_lstm" and neigh is not None and neigh_mask is not None:
            for k in range(neigh.shape[0]):
                if neigh_mask[k] > 0.5:
                    ax.plot(
                        neigh[k, :, 0],
                        neigh[k, :, 1],
                        color="lightgray",
                        linewidth=1.0,
                        alpha=0.9,
                    )

        past_show = min(frame + 1, args.past_len)
        ax.plot(
            ego[:past_show, 0],
            ego[:past_show, 1],
            "o-",
            color="black",
            linewidth=2.0,
            markersize=3,
            label="past",
        )

        if frame >= args.past_len:
            f_show = min(frame - args.past_len + 1, args.future_len)
            ax.plot(
                tgt[:f_show, 0],
                tgt[:f_show, 1],
                "o-",
                color="tab:green",
                linewidth=2.0,
                markersize=3,
                label="future_gt",
            )
            ax.plot(
                pred[:f_show, 0],
                pred[:f_show, 1],
                "o--",
                color="tab:red",
                linewidth=2.0,
                markersize=3,
                label="future_pred",
            )

        ax.scatter([0.0], [0.0], color="tab:blue", s=35)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_aspect("equal", adjustable="box")
        if not args.clean:
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("x_local (m)")
            ax.set_ylabel("y_local (m)")
            ax.set_title(
                "Model rollout | {} | sample {} | t={}/{}".format(
                    model_type, si, frame + 1, total_frames
                )
            )
            ax.legend(loc="best", fontsize=8)
        else:
            ax.set_axis_off()
        return []

    anim = FuncAnimation(fig, draw, frames=total_frames, interval=int(1000 / max(1, args.fps)))
    out_gif = os.path.join(args.out_dir, "model_rollout_{}_sample{:04d}.gif".format(model_type, si))
    anim.save(out_gif, writer=PillowWriter(fps=max(1, args.fps)))
    plt.close(fig)

    print("model_type: {}".format(model_type))
    print("sample_idx: {}".format(si))
    print("gif: {}".format(out_gif))


if __name__ == "__main__":
    main()
