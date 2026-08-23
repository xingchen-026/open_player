"""World-model rollout helpers for the planner."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from open_player.core.schema import SchemaSet
from open_player.core.types import Goal, WorldState
from open_player.world.model import Prediction, WorldModel


class WorldModelRollout:
    """Short latent rollouts over candidate action sequences."""

    def __init__(self, model: Optional[WorldModel], schema: SchemaSet, horizon: int = 4) -> None:
        self.model = model
        self.schema = schema
        self.horizon = int(horizon)

    def available(self) -> bool:
        return self.model is not None

    def rollout(self, state: WorldState, actions: List[int]) -> List[Prediction]:
        if self.model is None:
            return []
        return self.model.rollout(state, actions, k=len(actions))

    def evaluate(self, state: WorldState, predictions: List[Prediction], goal: Goal) -> float:
        """Heuristic utility of a rollout with respect to a goal (in [-1, 1])."""
        if not predictions:
            return 0.0
        last = predictions[-1]
        sp = last.spatial_pred[0].detach().cpu().numpy()
        if goal.goal_type == "exploration" or goal.goal_type == "information":
            novelty_gain = float(sp[3].mean()) if sp.shape[0] > 3 else 0.0
            return min(max(novelty_gain, -1.0), 1.0)
        if goal.goal_type == "survival":
            threat = float(sp[2].mean()) if sp.shape[0] > 2 else 0.0
            return -min(threat, 1.0)
        # task / learning / skill_improvement: proximity to target entities
        if goal.target:
            ents = last.predicted_entity_states(self.schema, batch=0)
            targets = [e for e in ents if e.semantic_type == goal.target]
            player = next((e for e in ents if e.semantic_type == "player"), None)
            if targets and player is not None:
                nearest = min(targets, key=lambda e: float(np.linalg.norm(np.asarray(e.position) - np.asarray(player.position))))
                d = float(np.linalg.norm(np.asarray(nearest.position) - np.asarray(player.position)))
                return max(-1.0, min(1.0, 1.0 / (1.0 + d)))
        return 0.0
