import torch
import torch.nn as nn


class MultimodalLSTM(nn.Module):
    """
    Single-agent LSTM encoder + K independent trajectory heads (multimodal).

    Forward returns [B, K, F, 2] — use winner-take-all SmoothL1 in training
    and minADE@K / minFDE@K style metrics at evaluation.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        future_len: int = 20,
        num_modes: int = 6,
    ):
        super().__init__()
        self.future_len = future_len
        self.num_modes = num_modes
        self.encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, future_len * 2),
                )
                for _ in range(num_modes)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        z = h[-1]  # [B, H]
        outs = []
        for head in self.heads:
            outs.append(head(z).view(-1, self.future_len, 2))
        return torch.stack(outs, dim=1)  # [B, K, F, 2]


def multimodal_wta_smooth_l1(pred_modes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred_modes: [B, K, F, 2], target: [B, F, 2]
    Per-sample min over modes of mean SmoothL1 over (F,2), then mean over batch.
    """
    y = target.unsqueeze(1).expand_as(pred_modes)
    try:
        diff = torch.nn.functional.smooth_l1_loss(pred_modes, y, reduction="none", beta=1.0)  # [B,K,F,2]
    except TypeError:
        diff = torch.nn.functional.smooth_l1_loss(pred_modes, y, reduction="none")  # [B,K,F,2]
    per_mode = diff.mean(dim=(2, 3))  # [B, K]
    return per_mode.min(dim=1)[0].mean()
