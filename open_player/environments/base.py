"""Environment base class (mirrors the Environment protocol in core.specs)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from open_player.core.types import Action, Observation


class Environment(ABC):
    """Minimal Gym-style environment interface, dependency-free."""

    @property
    def action_space(self) -> Any:
        """Action space (DiscreteActionSpace in Phase 0)."""
        return getattr(self, "_action_space", None)

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action | int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        ...
