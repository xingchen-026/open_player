"""Interface contracts (protocols) shared across layers.

Concrete modules depend on these protocols, never on each other's internal
implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Tuple

from open_player.core.types import Action, Observation, WorldState


class Environment(Protocol):
    """Minimal environment contract (Gym-style but dependency-free)."""

    @property
    def action_space(self) -> Any:  # pragma: no cover
        """DiscreteActionSpace (or continuous/hybrid in later phases)."""
        ...

    def reset(self, seed: int | None = None) -> Observation:
        """Start a new episode, return the initial observation."""
        ...

    def step(self, action: Action | int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Apply one action, return (observation, reward, done, info)."""
        ...


class ObservationEncoder(ABC):
    """Turns an environment Observation into a WorldState.

    Phase 0 ships DummyVisionEncoder (structured observation -> WorldState).
    A real vision encoder will later implement the same interface.
    """

    @abstractmethod
    def encode(self, observation: Observation, t: int = 0) -> WorldState:
        """Encode an observation into a (single-batch) WorldState."""
        ...
