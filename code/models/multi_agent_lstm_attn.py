import torch
import torch.nn as nn


class MultiAgentLSTMAttention(nn.Module):
    """
    Multi-agent LSTM with ego-query cross-attention over neighbor encodings
    (replaces masked mean pooling).

    Inputs:
      ego_past: [B, P, 2]
      neigh_past: [B, K, P, 2]
      neigh_mask: [B, K] — 1 = valid neighbor slot
    Output:
      pred: [B, F, 2]
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        neighbor_hidden_dim: int = 96,
        future_len: int = 20,
        attn_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        self.ego_encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )
        self.neighbor_encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=neighbor_hidden_dim,
            batch_first=True,
            num_layers=1,
        )
        self.q_proj = nn.Linear(hidden_dim, attn_dim)
        self.kv_proj = nn.Linear(neighbor_hidden_dim, attn_dim)
        self.cross_attn = nn.MultiheadAttention(
            attn_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        fused_dim = hidden_dim + attn_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_len * 2),
        )

    def forward(
        self,
        ego_past: torch.Tensor,
        neigh_past: torch.Tensor,
        neigh_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, (h_ego, _) = self.ego_encoder(ego_past)
        z_ego = h_ego[-1]  # [B, H]

        b, k, p, d = neigh_past.shape
        neigh_flat = neigh_past.reshape(b * k, p, d)
        _, (h_neigh, _) = self.neighbor_encoder(neigh_flat)
        z_neigh = h_neigh[-1].reshape(b, k, -1)  # [B, K, Hn]

        q = self.q_proj(z_ego).unsqueeze(1)  # [B, 1, D]
        kv = self.kv_proj(z_neigh)  # [B, K, D]
        # PyTorch: True = ignore position
        pad = neigh_mask <= 0.5
        ctx, _ = self.cross_attn(q, kv, kv, key_padding_mask=pad)
        ctx = ctx.squeeze(1)  # [B, D]

        z = torch.cat([z_ego, ctx], dim=-1)
        out = self.head(z)
        return out.view(b, self.future_len, 2)
