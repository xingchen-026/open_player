"""Action spaces and specs.

The Action interface must support discrete, continuous and hybrid spaces.
Phase 0 fully implements the discrete space; the continuous/hybrid classes
exist as stable interfaces for later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class DiscreteActionSpace:
    """A discrete action space: an ordered list of named actions."""

    def __init__(self, names: Sequence[str]) -> None:
        if not names:
            raise ValueError("DiscreteActionSpace needs at least one action")
        self.names: List[str] = list(names)
        self._index = {name: i for i, name in enumerate(self.names)}

    @property
    def n(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        if name not in self._index:
            raise KeyError(f"unknown action '{name}'; available: {self.names}")
        return self._index[name]

    def name(self, index: int) -> str:
        return self.names[int(index)]

    def contains(self, name: str) -> bool:
        return name in self._index

    def sample(self, rng: Optional[np.random.Generator] = None) -> int:
        if rng is not None:
            return int(rng.integers(0, self.n))
        return int(np.random.randint(0, self.n))

    def to_spec(self) -> "ActionSpec":
        return ActionSpec(kind="discrete", action_names=list(self.names))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:  # pragma: no cover
        return f"DiscreteActionSpace({self.names})"


class ContinuousActionSpace:
    """A continuous action space (interface reserved for later phases)."""

    def __init__(self, bounds: Sequence[Tuple[float, float]]) -> None:
        self.bounds: List[Tuple[float, float]] = [tuple(b) for b in bounds]

    @property
    def dim(self) -> int:
        return len(self.bounds)

    def sample(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is not None:
            return np.asarray([rng.uniform(lo, hi) for lo, hi in self.bounds], dtype=np.float32)
        return np.asarray([np.random.uniform(lo, hi) for lo, hi in self.bounds], dtype=np.float32)


class HybridActionSpace:
    """Discrete + continuous parts (interface reserved for later phases)."""

    def __init__(self, discrete: DiscreteActionSpace, continuous: ContinuousActionSpace) -> None:
        self.discrete = discrete
        self.continuous = continuous


@dataclass
class ActionSpec:
    """Serialisable description of an action interface."""

    kind: str  # discrete | continuous | hybrid
    action_names: List[str] = field(default_factory=list)
    continuous_bounds: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "action_names": list(self.action_names),
            "continuous_bounds": [list(b) for b in self.continuous_bounds],
        }
