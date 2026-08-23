"""DynamicsModel: multi-step latent dynamics (residual MLP).

z_{t+1} = z_t + f(z_t, action_t).  The residual form keeps 1-step learning
stable and the interface ready for multi-step rollout.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class DynamicsModel(nn.Module):
    """Latent transition model with an explicit action input."""

    def __init__(self, num_actions: int, latent_dim: int, hidden: int = 128, action_embedding_dim: int = 32) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.latent_dim = int(latent_dim)
        self.action_embedding = nn.Embedding(self.num_actions, int(action_embedding_dim))
        self.net = nn.Sequential(
            nn.Linear(latent_dim + int(action_embedding_dim), int(hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden), latent_dim),
        )

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        """action: [B] long indices (or [B, num_actions] float vector)."""
        if action.dim() == 2 and action.dtype.is_floating_point:
            a = action
        else:
            a = self.action_embedding(action.to(dtype=torch.long))
        delta = self.net(torch.cat([z, a], dim=-1))
        return z + delta
