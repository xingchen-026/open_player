"""SkillRegistry: named skills + candidate generation for the planner."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from open_player.actions.controller import ActionController
from open_player.core.types import Goal, WorldState
from open_player.skills.base import Skill
from open_player.skills.rule import ApproachSkill, AvoidThreatSkill, CollectSkill, ExploreSkill


class SkillRegistry:
    """Holds skill instances; the planner queries candidates from here."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> Skill:
        self._skills[skill.name] = skill
        return skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"unknown skill '{name}'; available: {sorted(self._skills)}")
        return self._skills[name]

    def names(self) -> List[str]:
        return sorted(self._skills)

    def candidates(self, state: WorldState, goal: Optional[Goal] = None) -> List[Skill]:
        return [s for s in self._skills.values() if s.can_start(state)]

    def reset_all(self) -> None:
        for s in self._skills.values():
            s.reset()

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    @classmethod
    def build_default(cls, controller: ActionController, config: Any) -> "SkillRegistry":
        """The Phase 0 default skill set (short/medium/long horizons)."""
        horizons = config.get("planning.horizons", {"short": 4, "medium": 8, "long": 32})
        reg = cls()
        reg.register(ExploreSkill(controller, horizon=int(horizons["medium"]), name="explore"))
        reg.register(ApproachSkill(controller, target_type="resource", horizon=int(horizons["short"]), name="approach_resource"))
        reg.register(CollectSkill(controller, horizon=int(horizons["short"]), name="collect"))
        reg.register(AvoidThreatSkill(controller, horizon=int(horizons["short"]), name="avoid_threat"))
        return reg
