"""Planner: hierarchical model-based planner (heuristic in Phase 0).

Goal -> Subgoal -> Candidate Skills -> World Model Rollout -> Outcome
Evaluation -> Select Skill.

No MCTS, no large search in Phase 0; the interface is designed so a learned
planner can replace the heuristic one later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from open_player.core.schema import SchemaSet
from open_player.core.types import Goal, WorldState
from open_player.planning.rollout import WorldModelRollout
from open_player.planning.scoring import OutcomeEvaluator
from open_player.skills.base import Skill
from open_player.skills.registry import SkillRegistry
from open_player.world.model import WorldModel


@dataclass
class Plan:
    """The planner's output for one goal."""

    goal: Goal
    skill_name: str
    skill: Skill
    horizon: int
    subgoals: List[str] = field(default_factory=list)
    expected_utility: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)


class Planner:
    """Selects a skill for the current goal using short world-model rollouts."""

    def __init__(
        self,
        config: Any,
        registry: SkillRegistry,
        schema: SchemaSet,
        model: Optional[WorldModel] = None,
        procedural_memory: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.schema = schema
        self.horizons: Dict[str, int] = {k: int(v) for k, v in config.get("planning.horizons", {}).items()}
        self.goal_type_horizons: Dict[str, str] = dict(config.get("planning.goal_type_horizons", {}))
        self.max_candidates = int(config.get("planning.max_candidates", 4))
        self.rollout_steps = int(config.get("planning.rollout_steps", 4))
        self.rollout = WorldModelRollout(model, schema, horizon=self.rollout_steps)
        self.evaluator = OutcomeEvaluator(config, procedural_memory=procedural_memory)

    def horizon_for(self, goal: Goal) -> int:
        tier = self.goal_type_horizons.get(goal.goal_type, "short")
        return int(self.horizons.get(tier, 4))

    def plan(self, state: WorldState, goal: Goal) -> Plan:
        horizon = self.horizon_for(goal)
        candidates = self.registry.candidates(state, goal)
        if not candidates:
            candidates = [self.registry.get("explore")]

        # static scoring first, then world-model rollout only for the top-2
        selected_candidates = candidates[: self.max_candidates]
        static_scores = self.evaluator.score_all(goal, selected_candidates, state, {})
        rollout_utilities: Dict[str, float] = {}
        if self.rollout.available() and selected_candidates:
            top_names = sorted(static_scores, key=static_scores.get, reverse=True)[:2]
            for skill in [s for s in selected_candidates if s.name in top_names]:
                try:
                    first_action = skill.act(state)
                except Exception:  # pragma: no cover
                    first_action = None
                if first_action is not None:
                    actions = [first_action.index] * self.rollout_steps
                    preds = self.rollout.rollout(state, actions)
                    rollout_utilities[skill.name] = self.rollout.evaluate(state, preds, goal)
                skill.reset()

        scores = self.evaluator.score_all(goal, selected_candidates, state, rollout_utilities)
        best_name = max(scores, key=scores.get)
        skill = self.registry.get(best_name)
        skill.reset()
        return Plan(
            goal=goal,
            skill_name=best_name,
            skill=skill,
            horizon=horizon,
            subgoals=[f"execute {best_name} for up to {horizon} steps"],
            expected_utility=float(scores[best_name]),
            scores={k: float(v) for k, v in scores.items()},
        )
