"""Goal system: candidates -> utility scoring -> selection -> progress.

External Goal + Intrinsic Motivation + Current Situation
    -> Candidate Goals -> Goal Scoring -> Selected Goal

Phase 0 uses a utility-based scorer (no learned scorer yet).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from open_player.core.types import EntityState, Goal, GoalType, WorldState


class UtilityGoalScorer:
    """Utility = type weight * priority + situation bonuses."""

    def __init__(self, config: Any) -> None:
        self.type_weights: Dict[str, float] = dict(config.get("goals.type_weights", {}))
        self._counter = 0

    def score(self, goal: Goal, state: WorldState, motivation: Dict[str, float]) -> float:
        w = float(self.type_weights.get(goal.goal_type, 0.5))
        score = w * float(goal.priority)
        # proximity bonus for located goals
        if goal.location is not None:
            player = self._player(state)
            if player is not None:
                dist = float(np.linalg.norm(np.asarray(goal.location, dtype=np.float32) - np.asarray(player.position, dtype=np.float32)))
                score += 0.5 / (1.0 + dist)
        # drive + situational bonuses
        if goal.goal_type == GoalType.EXPLORATION.value:
            score += 0.3 * motivation.get("novelty", 0.0)
        elif goal.goal_type == GoalType.SURVIVAL.value:
            threat = motivation.get("threat", 0.0)
            score += 0.3 * threat + 0.8 * threat
        elif goal.goal_type == GoalType.LEARNING.value:
            score += 0.3 * motivation.get("curiosity", 0.0)
        elif goal.goal_type == GoalType.TASK.value:
            # a visible, concrete target is a strong objective
            score += 0.8
        return score

    @staticmethod
    def _player(state: WorldState) -> Optional[EntityState]:
        for e in state.entity_states(0):
            if e.semantic_type == "player":
                return e
        return None


class GoalManager:
    """Generates, scores and tracks goals."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.scorer = UtilityGoalScorer(config)
        self._counter = 0
        self.goal_min_exploration = float(config.get("training.goal_min_exploration", 0.6))
        self.novelty_threshold = float(config.get("goals.novelty_threshold", 0.2))
        self.threat_threshold = float(config.get("goals.threat_threshold", 0.5))
        self.learning_threshold = float(config.get("goals.learning_threshold", 0.05))

    # ------------------------------------------------------------------ #
    def generate_candidates(
        self, state: WorldState, motivation: Dict[str, float], env_info: Dict[str, Any]
    ) -> List[Goal]:
        entities = state.entity_states(0)
        player = next((e for e in entities if e.semantic_type == "player"), None)
        candidates: List[Goal] = []
        t = int(state.t)

        # task: collect a visible resource
        resources = [e for e in entities if e.semantic_type == "resource"]
        if resources and player is not None:
            nearest = min(resources, key=lambda e: float(np.linalg.norm(np.asarray(e.position) - np.asarray(player.position))))
            dist = float(np.linalg.norm(np.asarray(nearest.position) - np.asarray(player.position)))
            candidates.append(
                Goal(
                    goal_id=self._id("task"), goal_type=GoalType.TASK.value, target="resource",
                    location=np.asarray(nearest.position, dtype=np.float32).copy(),
                    priority=min(1.0, 0.5 + 0.5 / (1.0 + dist)), source="external", created_t=t,
                    metadata={"entity_id": nearest.entity_id},
                )
            )

        # survival: threat is high
        threat = float(env_info.get("threat_level", 0.0))
        if threat >= self.threat_threshold:
            candidates.append(
                Goal(goal_id=self._id("survival"), goal_type=GoalType.SURVIVAL.value, target="player",
                     priority=min(1.0, threat), source="intrinsic:survival", created_t=t)
            )

        # exploration: world still novel (Phase 1: intrinsic novelty reward
        # also contributes to the exploration priority)
        if motivation.get("novelty", 0.0) >= self.novelty_threshold:
            bonus = float(env_info.get("intrinsic_novelty", 0.0)) * 0.3
            candidates.append(
                Goal(goal_id=self._id("exploration"), goal_type=GoalType.EXPLORATION.value, target="unknown",
                     priority=min(1.0, motivation["novelty"] + 0.3 + bonus), source="intrinsic:novelty", created_t=t)
            )

        # learning: world model error is high
        if motivation.get("curiosity", 0.0) >= self.learning_threshold:
            candidates.append(
                Goal(goal_id=self._id("learning"), goal_type=GoalType.LEARNING.value, target="world_model",
                     priority=min(1.0, motivation["curiosity"] + 0.3), source="intrinsic:curiosity", created_t=t)
            )

        # information: unknown area remains; Phase 1: high world-model
        # uncertainty raises the information-goal priority (epistemic drive)
        if motivation.get("unknown", 0.0) > 0.05 or env_info.get("uncertainty_mean", 0.0) > 0.0:
            info_bonus = float(self.config.get("intrinsic.info_goal_bonus", 0.0))
            uncertainty = float(env_info.get("uncertainty_mean", 0.0))
            candidates.append(
                Goal(goal_id=self._id("information"), goal_type=GoalType.INFORMATION.value, target="unknown",
                     priority=min(1.0, 0.4 * motivation["unknown"] + 0.1 + info_bonus * uncertainty),
                     source="intrinsic:information", created_t=t)
            )

        if not candidates:
            candidates.append(
                Goal(goal_id=self._id("exploration"), goal_type=GoalType.EXPLORATION.value, target="unknown",
                     priority=0.4, source="intrinsic:novelty", created_t=t)
            )
        return candidates

    def select(
        self,
        state: WorldState,
        motivation: Dict[str, float],
        env_info: Dict[str, Any],
        force_type: Optional[str] = None,
    ) -> Goal:
        candidates = self.generate_candidates(state, motivation, env_info)
        if force_type is not None:
            forced = [g for g in candidates if g.goal_type == force_type]
            if forced:
                candidates = forced
        scored = sorted(candidates, key=lambda g: self.scorer.score(g, state, motivation), reverse=True)
        return scored[0]

    def update(self, goal: Goal, state: WorldState, env_info: Dict[str, Any]) -> str:
        """Update progress; return the new status."""
        gtype = goal.goal_type
        if gtype == GoalType.TASK.value:
            goal.progress = float(env_info.get("collected", 0)) / max(1.0, float(env_info.get("num_resources", 1)))
            if env_info.get("collected_this_step"):
                goal.status = "succeeded"
        elif gtype == GoalType.EXPLORATION.value:
            sp = state.spatial_t[0].detach().cpu().numpy()
            visited = float(sp[7].mean()) if sp.shape[0] > 7 else 0.0
            goal.progress = visited
            if visited >= self.goal_min_exploration:
                goal.status = "succeeded"
        elif gtype == GoalType.SURVIVAL.value:
            goal.progress = float(env_info.get("hp", 1)) / max(1.0, float(env_info.get("hp_max", 3)))
            goal.metadata["safe_steps"] = goal.metadata.get("safe_steps", 0) + (0 if env_info.get("hp_delta", 0) < 0 else 1)
            if goal.metadata["safe_steps"] >= 8:
                goal.status = "succeeded"
            if env_info.get("hp", 1) <= 0:
                goal.status = "failed"
        elif gtype == GoalType.LEARNING.value:
            err = float(env_info.get("world_model_error", 1.0))
            goal.progress = max(0.0, 1.0 - err)
            if err <= self.learning_threshold:
                goal.status = "succeeded"
        else:
            goal.progress = min(goal.progress + 0.05, 1.0)
        return goal.status

    def _id(self, prefix: str) -> str:
        self._counter += 1
        return f"goal-{prefix}-{self._counter}"
