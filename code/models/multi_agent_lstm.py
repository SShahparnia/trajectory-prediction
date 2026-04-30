import torch
import torch.nn as nn


class MultiAgentLSTM(nn.Module):
    """
    Inputs:
      ego_past: [B, P, 2]
      neigh_past: [B, K, P, 2]
      neigh_mask: [B, K] where 1 means valid neighbor, 0 means padded
    Output:
      pred: [B, F, 2]
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        neighbor_hidden_dim: int = 96,
        future_len: int = 20,
    ):
        super().__init__()
        self.future_len = future_len
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
        fused_dim = hidden_dim + neighbor_hidden_dim
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
        neigh_flat = neigh_past.view(b * k, p, d)
        _, (h_neigh, _) = self.neighbor_encoder(neigh_flat)
        z_neigh = h_neigh[-1].view(b, k, -1)  # [B, K, Hn]

        mask = neigh_mask.unsqueeze(-1).float()  # [B, K, 1]
        z_neigh = z_neigh * mask
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        z_neigh_pool = z_neigh.sum(dim=1) / denom

        z = torch.cat([z_ego, z_neigh_pool], dim=-1)
        out = self.head(z)
        return out.view(b, self.future_len, 2)
