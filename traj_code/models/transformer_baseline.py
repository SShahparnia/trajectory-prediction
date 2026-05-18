import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        t = x.shape[1]
        return x + self.pe[:, :t, :]


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        future_len: int = 20,
    ):
        super().__init__()
        self.future_len = future_len
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model=d_model, max_len=256)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, future_len * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, past_len, 2]
        h = self.in_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        out = self.head(h)
        return out.view(x.shape[0], self.future_len, 2)
