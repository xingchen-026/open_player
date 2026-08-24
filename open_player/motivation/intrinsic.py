"""Intrinsic reward for Phase 1 exploration.

    r_intrinsic = alpha * prediction_error
                + beta  * novelty * decay^visits
                - risk_penalty * threat_at_player
                - repetition_penalty * repeated_action
                + gamma * information_gain

All weights come from config.intrinsic.  The signal must serve information
acquisition, not "novel for the sake of novel": novelty decays with visit
counts, risky cells (threat) are penalised, and repeated actions are
penalised.  The reward is consumed by goal selection (GoalManager) and by
the ExploreSkill's target choice, not just logged.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from open_player.core.state import structured_grid
from open_player.core.types import WorldState


class VisitCounter:
    """Per-grid-cell visit counts (novelty decay support)."""

    def __init__(self, cap: int = 10) -> None:
        self.cap = int(cap)
        self.counts: Dict[tuple, float] = {}

    def update(self, pos: np.ndarray) -> None:
        key = (int(round(float(pos[0]))), int(round(float(pos[1]))))
        self.counts[key] = min(self.counts.get(key, 0.0) + 1.0, float(self.cap))

    def get(self, pos: Optional[np.ndarray]) -> float:
        if pos is None:
            return 0.0
        key = (int(round(float(pos[0]))), int(round(float(pos[1]))))
        return float(self.counts.get(key, 0.0))

    def decay_all(self, decay: float) -> None:
        self.counts = {k: v * float(decay) for k, v in self.counts.items()}

    def reset(self) -> None:
        self.counts.clear()

    def stats(self) -> Dict[str, Any]:
        return {"num_visited_cells": len(self.counts), "mean_visits": float(np.mean(list(self.counts.values()))) if self.counts else 0.0}


class IntrinsicReward:
    """Computes the Phase 1 intrinsic reward and exploration utility maps."""

    def __init__(self, config: Any) -> None:
        ic = config.intrinsic if hasattr(config, "intrinsic") else config.get("intrinsic", {})
        self.alpha = float(ic.get("alpha", 1.0))
        self.beta = float(ic.get("beta", 0.3))
        self.gamma = float(ic.get("gamma", 0.2))
        self.risk_penalty = float(ic.get("risk_penalty", 0.5))
        self.repetition_penalty = float(ic.get("repetition_penalty", 0.2))
        self.novelty_decay = float(ic.get("novelty_decay", 0.99))
        self.visit_count_cap = int(ic.get("visit_count_cap", 10))

    # ------------------------------------------------------------------ #
    def compute(
        self,
        *,
        state: WorldState,
        world_model_error: float = 0.0,
        uncertainty_mean: float = 0.0,
        prev_uncertainty_mean: Optional[float] = None,
        action: Optional[int] = None,
        prev_action: Optional[int] = None,
        visit_counter: Optional[VisitCounter] = None,
        player_pos: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute the intrinsic reward and its components."""
        novelty = self._novelty(state)
        visits = 0.0 if visit_counter is None else visit_counter.get(player_pos)
        novelty_term = self.beta * novelty * (self.novelty_decay ** visits)
        err_norm = min(max(float(world_model_error), 0.0) / 10.0, 1.0)
        err_term = self.alpha * err_norm
        ig = 0.0 if prev_uncertainty_mean is None else abs(float(prev_uncertainty_mean) - float(uncertainty_mean))
        ig_term = self.gamma * min(ig, 1.0)
        risk = self._risk_at_player(state, player_pos)
        risk_term = self.risk_penalty * risk
        rep = 1.0 if (action is not None and prev_action is not None and action == prev_action) else 0.0
        rep_term = self.repetition_penalty * rep
        total = err_term + novelty_term + ig_term - risk_term - rep_term
        return {
            "total": float(total),
            "prediction_error": float(err_term),
            "novelty": float(novelty_term),
            "information_gain": float(ig_term),
            "risk": float(risk_term),
            "repetition": float(rep_term),
            "visits": float(visits),
        }

    # ------------------------------------------------------------------ #
    def explore_utility_map(self, state: WorldState, visit_counter: Optional[VisitCounter] = None) -> np.ndarray:
        """Per-cell intrinsic utility for exploration target selection.

        utility = novelty * decay^visits - risk_penalty * threat
        Walls get -inf so they are never chosen.
        """
        novelty = structured_grid(state, "novelty")
        threat = structured_grid(state, "threat")
        wall = structured_grid(state, "wall")
        gs = novelty.shape[0]
        visits = np.zeros((gs, gs), dtype=np.float32)
        if visit_counter is not None:
            for (x, y), v in visit_counter.counts.items():
                if 0 <= x < gs and 0 <= y < gs:
                    visits[y, x] = v
        util = novelty * (self.novelty_decay ** visits) - self.risk_penalty * threat
        util[wall > 0.5] = -np.inf
        return util

    # ------------------------------------------------------------------ #
    @staticmethod
    def _player(state: WorldState) -> Optional[Any]:
        for e in state.entity_states(0):
            if e.semantic_type == "player":
                return e
        return None

    @staticmethod
    def _novelty(state: WorldState) -> float:
        try:
            return float(structured_grid(state, "novelty").mean())
        except Exception:  # pragma: no cover - defensive
            return 1.0

    @staticmethod
    def _risk_at_player(state: WorldState, player_pos: Optional[np.ndarray]) -> float:
        try:
            threat = structured_grid(state, "threat")
            pos = player_pos
            if pos is None:
                p = IntrinsicReward._player(state)
                if p is None:
                    return 0.0
                pos = p.position
            gx, gy = int(round(float(pos[0]))), int(round(float(pos[1])))
            if 0 <= gx < threat.shape[1] and 0 <= gy < threat.shape[0]:
                return float(threat[gy, gx])
        except Exception:  # pragma: no cover - defensive
            pass
        return 0.0
