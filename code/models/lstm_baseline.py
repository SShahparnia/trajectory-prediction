import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, future_len: int = 20):
        super().__init__()
        self.future_len = future_len
        self.encoder = nn.LSTM(
            input_size=in_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_len * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        z = h[-1]
        out = self.head(z)
        return out.view(x.shape[0], self.future_len, 2)
