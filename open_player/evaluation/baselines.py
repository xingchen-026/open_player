"""Baseline agents for Phase 1 comparisons: Random and Rule.

The Rule baseline is a hand-written policy (flee threats, chase resources,
explore novelty) using the same state decode path as Phase 0 skills.  It
carries NO learned components, so it is the honest heuristic reference.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from open_player.core.state import structured_grid
from open_player.core.types import Action, WorldState


class RandomBaseline:
    """Uniformly random action policy."""

    def __init__(self, action_space: Any, seed: int = 0) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def act(self, state: WorldState) -> Action:
        idx = self.action_space.sample(self.rng)
        return Action(name=self.action_space.name(idx), index=idx)


class RuleBaseline:
    """Hand-written rule policy (Phase 0 heuristics without any model)."""

    def __init__(self, action_space: Any, schema: Any, seed: int = 0) -> None:
        self.action_space = action_space
        self.schema = schema
        self.rng = np.random.default_rng(seed)
        self._names = list(action_space.names)
        self._idx = {n: i for i, n in enumerate(self._names)}

    def act(self, state: WorldState) -> Action:
        ents = [e for e in state.entity_states(0) if e.semantic_type != "empty"]
        player = next((e for e in ents if e.semantic_type == "player"), None)
        if player is None:
            return Action(self._names[0], 0)
        enemies = [e for e in ents if e.semantic_type == "enemy"]
        resources = [e for e in ents if e.semantic_type == "resource"]
        p = np.asarray(player.position, dtype=np.float32)

        # 1) flee visible threats
        if enemies:
            nearest = min(enemies, key=lambda e: float(np.linalg.norm(np.asarray(e.position) - p)))
            if float(np.linalg.norm(np.asarray(nearest.position) - p)) <= 3.0:
                away = p - np.asarray(nearest.position, dtype=np.float32)
                return self._dir(away)

        # 2) chase the nearest visible resource
        if resources:
            nearest = min(resources, key=lambda e: float(np.linalg.norm(np.asarray(e.position) - p)))
            d = float(np.linalg.norm(np.asarray(nearest.position) - p))
            if d < 0.5:
                if "collect" in self._idx:
                    return Action("collect", self._idx["collect"])
                return Action(self._names[0], 0)
            return self._dir(np.asarray(nearest.position, dtype=np.float32) - p)

        # 3) explore: nearest unvisited cell
        try:
            novelty = structured_grid(state, "novelty")
            wall = structured_grid(state, "wall")
            target = self._nearest_novel(p, novelty, wall)
            if target is not None:
                return self._dir(target - p)
        except Exception:  # pragma: no cover - defensive
            pass
        return Action(self._names[0], 0)

    # ------------------------------------------------------------------ #
    def _dir(self, delta: np.ndarray) -> Action:
        ax, ay = abs(float(delta[0])), abs(float(delta[1]))
        cands: List[str] = []
        if ax >= ay:
            cands.append("left" if delta[0] < 0 else "right")
            cands.append("up" if delta[1] < 0 else "down")
        else:
            cands.append("up" if delta[1] < 0 else "down")
            cands.append("left" if delta[0] < 0 else "right")
        for name in cands:
            if name in self._idx:
                return Action(name, self._idx[name])
        return Action(self._names[0], 0)

    @staticmethod
    def _nearest_novel(p: np.ndarray, novelty: np.ndarray, wall: np.ndarray) -> Optional[np.ndarray]:
        from collections import deque
        H, W = novelty.shape
        start = (int(round(float(p[0]))), int(round(float(p[1]))))
        if not (0 <= start[0] < W and 0 <= start[1] < H):
            return None
        q = deque([start])
        seen = {start}
        while q:
            x, y = q.popleft()
            if novelty[y, x] > 0.5 and wall[y, x] <= 0.5:
                return np.array([x, y], dtype=np.float32)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return None
