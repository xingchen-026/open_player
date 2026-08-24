"""LearnedChangePredictor: learned change / boundary signal (Phase 1).

Inputs: z_t, action_t, z_{t+1}.  Outputs:
* change logits [B, 2]  - probability of a meaningful world change
* boundary score [B, 1] - probability that t is an event boundary

No transformer, no event transformer: two-layer MLP only.  The Phase 0
HeuristicEventDetector stays untouched; the HybridEventDetector blends the
learned signal with heuristic evidence.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LearnedChangePredictor(nn.Module):
    """Predicts (change, boundary) from (z_t, action, z_t1)."""

    def __init__(self, latent_dim: int, num_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_embedding = nn.Embedding(int(num_actions), 16)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + 16, int(hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden), 3),
        )

    def forward(self, z_t: Tensor, action: Tensor, z_t1: Tensor) -> Tuple[Tensor, Tensor]:
        if action.dim() == 2 and action.dtype.is_floating_point:
            a = action
        else:
            a = self.action_embedding(action.to(dtype=torch.long))
        x = torch.cat([z_t, a, z_t1], dim=-1)
        out = self.net(x)  # [B, 3]
        change_logits = out[:, :2]
        boundary = torch.sigmoid(out[:, 2:3])
        return change_logits, boundary

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
