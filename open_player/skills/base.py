"""Skill interface (hierarchical skill / option).

A Skill has: initiation_condition, policy, termination_condition,
outcome_model, memory and metadata.  It is NOT a fixed action sequence and
it does NOT have to be a neural network (Phase 0 ships rule skills; the
NeuralSkill interface slot is reserved).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from open_player.core.types import Action, WorldState


@dataclass
class OutcomePrediction:
    """A skill's own model of what executing it will achieve."""

    skill_name: str
    expected_utility: float
    expected_events: List[str] = field(default_factory=list)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """Base class for all skills (rule-based, heuristic or neural)."""

    name: str = "skill"
    horizon: int = 4

    def __init__(self, name: str, horizon: int = 4) -> None:
        self.name = name
        self.horizon = int(horizon)
        self.memory: Dict[str, Any] = {}
        self._steps = 0

    # -- initiation ----------------------------------------------------- #
    @abstractmethod
    def can_start(self, state: WorldState) -> bool:
        """Initiation condition: can this skill run in this state?"""

    # -- policy --------------------------------------------------------- #
    @abstractmethod
    def act(self, state: WorldState, rng: Optional[np.random.Generator] = None) -> Action:
        """Pick one action under this skill's policy."""

    # -- termination ---------------------------------------------------- #
    @abstractmethod
    def should_terminate(self, state: WorldState) -> bool:
        """Termination condition."""

    # -- outcome model -------------------------------------------------- #
    @abstractmethod
    def predict_outcome(self, state: WorldState) -> OutcomePrediction:
        """The skill's own expected outcome (utility + events)."""

    # -- learning hook -------------------------------------------------- #
    def update(self, *, state: Optional[WorldState] = None, action: Optional[Action] = None, reward: float = 0.0, next_state: Optional[WorldState] = None, done: bool = False, **kwargs: Any) -> None:
        """Update the skill's memory / outcome model (no-op for rules)."""

    # -- lifecycle ------------------------------------------------------ #
    def reset(self) -> None:
        self._steps = 0

    @property
    def steps(self) -> int:
        return self._steps

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r}, horizon={self.horizon})"
