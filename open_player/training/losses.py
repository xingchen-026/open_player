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


def entity_prediction_loss(prediction: Prediction, target: WorldState, detach_target: bool = False) -> Tensor:
    """MSE over [B, N, D_entity], weighted by target existence probability.

    detach_target=True is used in Phase 1 (learned vision): the target
    features come from the same encoder and are treated as data, so only the
    prediction path carries gradient.
    """
    tgt = target.entities_t.detach() if detach_target else target.entities_t
    err = (prediction.entities_pred - tgt) ** 2
    weight = target.beliefs_t[:, :, 4:5].detach().clamp(min=0.0)  # existence
    return (err * weight).sum() / max(weight.sum(), 1.0)


def spatial_prediction_loss(prediction: Prediction, target: WorldState, detach_target: bool = False) -> Tensor:
    tgt = target.spatial_t.detach() if detach_target else target.spatial_t
    return torch.nn.functional.mse_loss(prediction.spatial_pred, tgt)


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
    detach_targets: bool = False,
) -> Dict[str, Tensor]:
    """All unweighted terms plus the weighted total ('total' key)."""
    w = dict(weights or {})
    losses: Dict[str, Tensor] = {}
    losses["entity"] = entity_prediction_loss(prediction, target_state, detach_target=detach_targets)
    losses["spatial"] = spatial_prediction_loss(prediction, target_state, detach_target=detach_targets)
    if change_label is not None:
        losses["change"] = change_prediction_loss(prediction, change_label)
    if target_z is not None:
        losses["latent"] = latent_consistency_loss(prediction, target_z)

    total = torch.zeros((), device=prediction.z_cur.device)
    for name, value in losses.items():
        total = total + float(w.get(name, 1.0)) * value
    losses["total"] = total
    return losses


def learned_change_loss(change_logits: torch.Tensor, change_label: torch.Tensor) -> torch.Tensor:
    """BCE for the LearnedChangePredictor (z_t, action, z_t1 -> change)."""
    return torch.nn.functional.binary_cross_entropy_with_logits(
        change_logits[:, 1], change_label.to(dtype=change_logits.dtype)
    )


def boundary_loss(boundary_score: torch.Tensor, boundary_label: torch.Tensor) -> torch.Tensor:
    """BCE for the learned boundary score (semantic events are boundaries)."""
    return torch.nn.functional.binary_cross_entropy(
        boundary_score.view(-1), boundary_label.to(dtype=boundary_score.dtype).view(-1)
    )


def spatial_variance_loss(spatial_features: torch.Tensor) -> torch.Tensor:
    """Anti-collapse regularizer for learned spatial features.

    Returns -mean(channel std over spatial dims): minimising it pushes the
    learned spatial representation away from a constant map (which would
    trivially satisfy the prediction loss but carry no information).
    """
    std = spatial_features.std(dim=(2, 3))
    return -std.mean()
