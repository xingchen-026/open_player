"""Non-learning world-model baselines (Phase 1.5 scientific controls).

* PersistenceWorldModel: predicts "nothing changes" (z_t1 = z_t, outputs =
  the current state's features).  Any learned model must beat this.
* RandomDynamicsWorldModel: a fixed random-initialised WorldModel that is
  never trained.  Controls for the inductive bias of the architecture.

Both expose the same prediction_errors() interface as WorldModel.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from open_player.core.schema import SchemaSet
from open_player.core.types import WorldState
from open_player.world.model import WorldModel


class PersistenceWorldModel:
    """Identity transition baseline (no parameters, no training)."""

    def __init__(self, schema: SchemaSet) -> None:
        self.schema = schema
        self._training = False

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def representation(self, state: WorldState) -> Any:
        class _Rep:
            pass
        rep = _Rep()
        # persistence "latent" = existence-weighted pooled entity features
        w = state.beliefs_t[:, :, 4:5].detach().clamp(min=0.0)
        rep.z = (state.entities_t.detach() * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return rep

    def num_parameters(self) -> int:
        return 0

    def prediction_errors(
        self,
        state: WorldState,
        actions: Sequence[int],
        target_states: List[WorldState],
        horizons: Sequence[int] = (1, 4, 8, 16),
    ) -> Dict[str, float]:
        """Persistence errors: compare the current state to each target."""
        errors: Dict[str, float] = {}
        z = self.representation(state).z
        ent = state.entities_t.detach()
        sp = state.spatial_t.detach()
        for k in horizons:
            if k > len(target_states):
                continue
            tgt = target_states[k - 1]
            errors[f"step{k}_entity"] = float(torch.nn.functional.mse_loss(ent, tgt.entities_t.detach()))
            errors[f"step{k}_spatial"] = float(torch.nn.functional.mse_loss(sp, tgt.spatial_t.detach()))
            tz = (tgt.entities_t.detach() * tgt.beliefs_t[:, :, 4:5].detach().clamp(min=0.0)).sum(dim=1) / tgt.beliefs_t[:, :, 4:5].detach().clamp(min=0.0).sum(dim=1).clamp(min=1.0)
            errors[f"step{k}_latent"] = float(torch.nn.functional.mse_loss(z, tz))
        errors["change_prob"] = 0.0
        errors["boundary_prob"] = 0.0
        return errors


class RandomDynamicsWorldModel:
    """Fixed random WorldModel (architecture control, never trained)."""

    def __init__(self, schema: SchemaSet, config: Any, num_actions: int, device: Any = "cpu") -> None:
        self.model = WorldModel(schema, config, num_actions=num_actions).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def representation(self, state: WorldState) -> Any:
        with torch.no_grad():
            return self.model.representation(state.detach())

    def num_parameters(self) -> int:
        return self.model.num_parameters()

    def prediction_errors(
        self,
        state: WorldState,
        actions: Sequence[int],
        target_states: List[WorldState],
        horizons: Sequence[int] = (1, 4, 8, 16),
    ) -> Dict[str, float]:
        with torch.no_grad():
            return self.model.prediction_errors(state.detach(), list(actions), [s.detach() for s in target_states], horizons=horizons)
