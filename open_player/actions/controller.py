"""ActionController: the single point where Actions meet an environment."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from open_player.actions.specs import DiscreteActionSpace
from open_player.core.types import Action

_DIRECTION_ACTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class ActionController:
    """Validates / converts Actions for one DiscreteActionSpace.

    Skills call direction_action(dx, dy) and receive a proper Action; the
    player passes the Action (or its index) to the environment.
    """

    def __init__(self, action_space: DiscreteActionSpace) -> None:
        self.space = action_space

    def to_index(self, action: Action | str | int) -> int:
        if isinstance(action, Action):
            return action.index
        if isinstance(action, str):
            return self.space.index(action)
        return int(action)

    def from_index(self, index: int) -> Action:
        return Action(name=self.space.name(index), index=int(index))

    def validate(self, action: Action | str | int) -> Action:
        idx = self.to_index(action)
        if not 0 <= idx < self.space.n:
            raise ValueError(f"action index {idx} out of range for space of size {self.space.n}")
        return Action(name=self.space.name(idx), index=idx)

    def direction_action(self, dx: float, dy: float, rng: Optional[np.random.Generator] = None) -> Action:
        """Pick the best available move action for a desired delta (dx, dy).

        Chooses the axis with the larger absolute delta; ties are broken
        randomly (or deterministically when no rng is given).
        """
        candidates: list[tuple[str, float]] = []
        ax, ay = abs(float(dx)), abs(float(dy))
        if ax >= ay and self.space.contains("left") and self.space.contains("right"):
            candidates.append(("left" if dx < 0 else "right", ax))
        if ay >= ax and self.space.contains("up") and self.space.contains("down"):
            candidates.append(("up" if dy < 0 else "down", ay))
        if not candidates:
            # Space has no move actions; fall back to a stable first action.
            return Action(name=self.space.name(0), index=0)
        candidates.sort(key=lambda item: item[1], reverse=True)
        top = [name for name, mag in candidates if mag >= candidates[0][1] - 1e-9]
        if rng is not None and len(top) > 1:
            name = top[int(rng.integers(0, len(top)))]
        else:
            name = top[0]
        return Action(name=name, index=self.space.index(name))

    def sample(self, rng: Optional[np.random.Generator] = None) -> Action:
        idx = self.space.sample(rng)
        return Action(name=self.space.name(idx), index=idx)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ActionController({self.space})"
