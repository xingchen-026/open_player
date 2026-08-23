"""Prediction losses for the self-supervised world model (Phase 0).

Minimum set required by the frozen plan:

* entity prediction loss (MSE, weighted by existence)
* spatial prediction loss (MSE)
* change prediction loss (BCE, optional when events are available)
* latent consistency loss (MSE between predicted and encoded target z)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from open_player.core.types import WorldState
from open_player.world.model import Prediction


def entity_prediction_loss(prediction: Prediction, target: WorldState) -> Tensor:
    """MSE over [B, N, D_entity], weighted by target existence probability."""
    err = (prediction.entities_pred - target.entities_t) ** 2
    weight = target.beliefs_t[:, :, 4:5].detach().clamp(min=0.0)  # existence
    return (err * weight).sum() / max(weight.sum(), 1.0)


def spatial_prediction_loss(prediction: Prediction, target: WorldState) -> Tensor:
    return torch.nn.functional.mse_loss(prediction.spatial_pred, target.spatial_t)


def change_prediction_loss(prediction: Prediction, change_label: Tensor) -> Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        prediction.change_logits[:, 1], change_label.to(dtype=prediction.change_logits.dtype)
    )


def latent_consistency_loss(prediction: Prediction, target_z: Tensor) -> Tensor:
    return torch.nn.functional.mse_loss(prediction.z_next, target_z)


def prediction_losses(
    prediction: Prediction,
    target_state: WorldState,
    change_label: Optional[Tensor] = None,
    target_z: Optional[Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Tensor]:
    """All unweighted terms plus the weighted total ('total' key)."""
    w = dict(weights or {})
    losses: Dict[str, Tensor] = {}
    losses["entity"] = entity_prediction_loss(prediction, target_state)
    losses["spatial"] = spatial_prediction_loss(prediction, target_state)
    if change_label is not None:
        losses["change"] = change_prediction_loss(prediction, change_label)
    if target_z is not None:
        losses["latent"] = latent_consistency_loss(prediction, target_z)

    total = torch.zeros((), device=prediction.z_cur.device)
    for name, value in losses.items():
        total = total + float(w.get(name, 1.0)) * value
    losses["total"] = total
    return losses
