"""Shared neural deconvolution head (3-layer MLP with BatchNorm/Dropout/softmax)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeconvHead(nn.Module):
    def __init__(self, input_dim: int, n_celltypes: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_celltypes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)


def train_deconv_head(
    feature_st: np.ndarray,
    pseudo_label: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float = 0.1,
    log_prefix: str = "[spaGPD]",
    device: torch.device | None = None,
) -> np.ndarray:
    from spagpd.io.utils import get_device, normalize_rows

    if device is None:
        device = get_device()
    x = torch.as_tensor(feature_st, dtype=torch.float32, device=device)
    target = torch.as_tensor(pseudo_label, dtype=torch.float32, device=device)
    model = DeconvHead(feature_st.shape[1], pseudo_label.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"{log_prefix} Train neural deconvolution head: epochs={epochs}", flush=True)
    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == epochs:
            entropy = -torch.sum(pred * torch.log(pred + 1e-8), dim=1).mean()
            print(
                f"{log_prefix} Deconv epoch={epoch:04d} mse={loss.item():.6f} entropy={entropy.item():.6f}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        return normalize_rows(model(x).cpu().numpy())
