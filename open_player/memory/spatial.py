"""SpatialMemoryStore: accumulated spatial memory across steps."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from open_player.core.types import WorldState


class SpatialMemoryStore:
    """Accumulates the spatial memory tensor over an episode.

    navigation memory = max over time of the navigation channel
    novelty = cells never occupied since the store was reset.
    """

    def __init__(self, shape: tuple, device: Any = "cpu") -> None:
        self.shape = tuple(int(s) for s in shape)
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.accum: Optional[np.ndarray] = None
        self.navigation: Optional[np.ndarray] = None
        self.ever_occupied: Optional[np.ndarray] = None

    def update(self, state: WorldState) -> None:
        sp = state.spatial_t[0].detach().cpu().numpy()
        if self.accum is None:
            self.accum = sp.copy()
        else:
            self.accum = np.maximum(self.accum, sp)
        # navigation (channel 5) accumulates via max; novelty = 1 - ever occupied (channel 0)
        nav = sp[5] if sp.shape[0] > 5 else np.zeros(sp.shape[1:], dtype=np.float32)
        occ = sp[0] if sp.shape[0] > 0 else np.zeros(sp.shape[1:], dtype=np.float32)
        self.navigation = nav if self.navigation is None else np.maximum(self.navigation, nav)
        self.ever_occupied = occ if self.ever_occupied is None else np.maximum(self.ever_occupied, occ)
        self.accum[5] = self.navigation
        if self.accum.shape[0] > 3:
            self.accum[3] = 1.0 - self.ever_occupied

    def get(self) -> np.ndarray:
        if self.accum is None:
            return np.zeros(self.shape, dtype=np.float32)
        return self.accum.copy()

    def novelty_mean(self) -> float:
        if self.accum is None or self.accum.shape[0] <= 3:
            return 1.0
        return float(self.accum[3].mean())

    def stats(self) -> Dict[str, Any]:
        return {"novelty_mean": self.novelty_mean()}
