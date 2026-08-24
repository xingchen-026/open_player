"""Outcome evaluation for candidate skills."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from open_player.core.types import Goal, WorldState
from open_player.skills.base import Skill


class OutcomeEvaluator:
    """Scores candidate skills for a goal.

    static affinity (goal-type -> preferred skills) + procedural success rate
    + optional world-model rollout utility.
    """

    DEFAULT_AFFINITY: Dict[str, Dict[str, float]] = {
        "task": {"collect": 1.0, "approach_resource": 0.8, "explore": 0.3, "avoid_threat": 0.1, "neural_explore": 0.3},
        "exploration": {"neural_explore": 1.0, "explore": 0.8, "approach_resource": 0.2, "avoid_threat": 0.2, "collect": 0.3},
        "survival": {"avoid_threat": 1.0, "explore": 0.4, "approach_resource": 0.1, "collect": 0.1, "neural_explore": 0.3},
        "learning": {"neural_explore": 0.9, "explore": 0.8, "approach_resource": 0.6, "collect": 0.5, "avoid_threat": 0.2},
        "information": {"neural_explore": 1.0, "explore": 0.9, "approach_resource": 0.3, "avoid_threat": 0.2, "collect": 0.2},
        "skill_improvement": {"collect": 0.7, "approach_resource": 0.7, "neural_explore": 0.7, "explore": 0.5, "avoid_threat": 0.3},
    }

    def __init__(self, config: Any, procedural_memory: Optional[Any] = None) -> None:
        self.config = config
        self.procedural = procedural_memory
        self.affinity = {k: dict(v) for k, v in self.DEFAULT_AFFINITY.items()}

    def score_skill(self, goal: Goal, skill: Skill, state: WorldState, rollout_utility: float = 0.0) -> float:
        table = self.affinity.get(goal.goal_type, {})
        base = float(table.get(skill.name, 0.2))
        pred = skill.predict_outcome(state)
        success = 0.5
        if self.procedural is not None:
            success = self.procedural.success_rate(skill.name, default=0.5)
        score = 0.5 * base + 0.25 * pred.expected_utility + 0.15 * success + 0.1 * rollout_utility
        # safety term: when a threat is near the player, prefer evasion
        from open_player.core.state import structured_grid
        threat_near = 0.0
        try:
            player = next((e for e in state.entity_states(0) if e.semantic_type == "player"), None)
            if player is not None:
                threat = structured_grid(state, "threat")
                gx = int(round(float(player.position[0])))
                gy = int(round(float(player.position[1])))
                if 0 <= gx < threat.shape[1] and 0 <= gy < threat.shape[0]:
                    threat_near = float(threat[gy, gx])
        except Exception:  # pragma: no cover - defensive
            threat_near = 0.0
        if threat_near > 0.5:
            if skill.name == "avoid_threat":
                score += 0.5
            elif skill.name in ("explore", "approach_resource", "collect"):
                score -= 0.4
        return float(score)

    def score_all(self, goal: Goal, skills: List[Skill], state: WorldState, rollout_utilities: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        rollout_utilities = rollout_utilities or {}
        return {s.name: self.score_skill(goal, s, state, rollout_utilities.get(s.name, 0.0)) for s in skills}
