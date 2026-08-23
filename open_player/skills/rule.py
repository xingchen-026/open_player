"""Rule / heuristic skills for Phase 0 (RuleSkill, HeuristicSkill bases)."""
from __future__ import annotations

from collections import deque
from typing import Any, List, Optional

import numpy as np

from open_player.actions.controller import ActionController
from open_player.core.state import grid_channel
from open_player.core.types import Action, EntityState, WorldState
from open_player.skills.base import OutcomePrediction, Skill


class RuleSkill(Skill):
    """Base class for rule-based skills with an action controller."""

    def __init__(self, name: str, controller: ActionController, horizon: int = 4) -> None:
        super().__init__(name=name, horizon=horizon)
        self.controller = controller
        self._done = False

    # -- helpers -------------------------------------------------------- #
    @staticmethod
    def _grid_channel(state: WorldState, channel: int) -> np.ndarray:
        """Exact grid-resolution view of a spatial memory channel."""
        return grid_channel(state, channel)

    @staticmethod
    def _player(state: WorldState) -> Optional[EntityState]:
        for e in state.entity_states(0):
            if e.semantic_type == "player":
                return e
        return None

    @staticmethod
    def _of_type(state: WorldState, semantic_type: str) -> List[EntityState]:
        return [e for e in state.entity_states(0) if e.semantic_type == semantic_type]

    @staticmethod
    def _spatial(state: WorldState, channel: int) -> np.ndarray:
        return state.spatial_t[0, channel].detach().cpu().numpy()

    def _direction(self, state: WorldState, target: np.ndarray, rng=None) -> Action:
        player = self._player(state)
        if player is None:
            return self.controller.from_index(0)
        delta = np.asarray(target, dtype=np.float32) - np.asarray(player.position, dtype=np.float32)
        return self.controller.direction_action(float(delta[0]), float(delta[1]), rng=rng)

    def _navigate(self, state: WorldState, target: np.ndarray, rng=None) -> Action:
        """Direction toward target, avoiding walls (fallback: any free move)."""
        player = self._player(state)
        if player is None:
            return self.controller.from_index(0)
        p = np.asarray(player.position, dtype=np.float32)
        delta = np.asarray(target, dtype=np.float32) - p
        wall = self._grid_channel(state, 1)
        H, W = wall.shape

        def free(name: str) -> bool:
            if not self.controller.space.contains(name):
                return False
            dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[name]
            gx, gy = int(round(p[0] + dx)), int(round(p[1] + dy))
            return 0 <= gx < W and 0 <= gy < H and wall[gy, gx] <= 0.5

        def try_axis(dx: float, dy: float) -> Optional[Action]:
            cands = []
            if abs(dx) >= abs(dy) and dx != 0:
                cands.append("left" if dx < 0 else "right")
                cands.append("up" if dy <= 0 else "down")
            elif dy != 0:
                cands.append("up" if dy < 0 else "down")
                cands.append("left" if dx <= 0 else "right")
            for name in cands:
                if free(name):
                    return Action(name=name, index=self.controller.space.index(name))
            return None

        primary = try_axis(float(delta[0]), float(delta[1]))
        if primary is not None:
            return primary
        # any free move as last resort
        for name in ("left", "right", "up", "down"):
            if free(name):
                return Action(name=name, index=self.controller.space.index(name))
        return self.controller.from_index(0)

    def _step_and_act(self, action: Action) -> Action:
        self._steps += 1
        return action

    def reset(self) -> None:
        super().reset()
        self._done = False

    def should_terminate(self, state: WorldState) -> bool:
        return self._done or self._steps >= self.horizon


class ExploreSkill(RuleSkill):
    """Move toward the nearest unvisited cell (novelty channel)."""

    def __init__(self, controller: ActionController, horizon: int = 8, name: str = "explore") -> None:
        super().__init__(name=name, controller=controller, horizon=horizon)

    def can_start(self, state: WorldState) -> bool:
        return True

    def act(self, state: WorldState, rng=None) -> Action:
        player = self._player(state)
        if player is None:
            return self._step_and_act(self.controller.from_index(0))
        novelty = self._grid_channel(state, 3)
        wall = self._grid_channel(state, 1)
        threat = self._grid_channel(state, 2)
        target = self._nearest_novel_cell(state, player, novelty, wall, threat)
        if target is None:
            self._done = True
            return self._step_and_act(self.controller.from_index(0))
        p = np.asarray(player.position, dtype=np.float32)
        if np.abs(p - target).sum() < 0.5:
            self._done = True
        return self._step_and_act(self._navigate(state, target, rng=rng))

    def should_terminate(self, state: WorldState) -> bool:
        return self._done or self._steps >= self.horizon

    def predict_outcome(self, state: WorldState) -> OutcomePrediction:
        novelty = float(self._spatial(state, 3).mean())
        return OutcomePrediction(self.name, expected_utility=novelty, expected_events=["move"], confidence=0.5)

    @staticmethod
    def _nearest_novel_cell(state: WorldState, player: EntityState, novelty: np.ndarray, wall: np.ndarray, threat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        # novelty / wall / threat are grid-resolution arrays; BFS in grid coords
        H, W = novelty.shape
        start = (int(round(float(player.position[0]))), int(round(float(player.position[1]))))
        if not (0 <= start[0] < W and 0 <= start[1] < H):
            return None
        if threat is None:
            threat = np.zeros_like(novelty)
        q: deque = deque([start])
        seen = {start}
        while q:
            x, y = q.popleft()
            if novelty[y, x] > 0.5 and wall[y, x] <= 0.5 and threat[y, x] <= 0.5:
                return np.array([x, y], dtype=np.float32)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen and threat[ny, nx] <= 0.5:
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return None


class ApproachSkill(RuleSkill):
    """Move toward the nearest visible entity of a semantic type."""

    def __init__(self, controller: ActionController, target_type: str, horizon: int = 4, name: Optional[str] = None) -> None:
        super().__init__(name=name or f"approach_{target_type}", controller=controller, horizon=horizon)
        self.target_type = target_type

    def can_start(self, state: WorldState) -> bool:
        return bool(self._of_type(state, self.target_type))

    def act(self, state: WorldState, rng=None) -> Action:
        targets = self._of_type(state, self.target_type)
        player = self._player(state)
        if not targets or player is None:
            self._done = True
            return self._step_and_act(self.controller.from_index(0))
        nearest = min(targets, key=lambda e: float(np.linalg.norm(np.asarray(e.position) - np.asarray(player.position))))
        d = float(np.linalg.norm(np.asarray(nearest.position) - np.asarray(player.position)))
        if d <= 1.5:
            self._done = True
            return self._step_and_act(self.controller.from_index(0))
        return self._step_and_act(self._navigate(state, np.asarray(nearest.position), rng=rng))

    def should_terminate(self, state: WorldState) -> bool:
        return self._done or self._steps >= self.horizon

    def predict_outcome(self, state: WorldState) -> OutcomePrediction:
        targets = self._of_type(state, self.target_type)
        player = self._player(state)
        if not targets or player is None:
            return OutcomePrediction(self.name, expected_utility=0.0, expected_events=[], confidence=0.0)
        nearest = min(targets, key=lambda e: float(np.linalg.norm(np.asarray(e.position) - np.asarray(player.position))))
        d = float(np.linalg.norm(np.asarray(nearest.position) - np.asarray(player.position)))
        return OutcomePrediction(self.name, expected_utility=1.0 / (1.0 + d), expected_events=["move"], confidence=0.6)


class CollectSkill(RuleSkill):
    """Approach a resource; collect it when adjacent."""

    def __init__(self, controller: ActionController, horizon: int = 8, name: str = "collect") -> None:
        super().__init__(name=name, controller=controller, horizon=horizon)
        self._collected = False

    def can_start(self, state: WorldState) -> bool:
        return bool(self._of_type(state, "resource"))

    def reset(self) -> None:
        super().reset()
        self._collected = False

    def act(self, state: WorldState, rng=None) -> Action:
        resources = self._of_type(state, "resource")
        player = self._player(state)
        if not resources or player is None:
            self._done = True
            return self._step_and_act(self.controller.from_index(0))
        nearest = min(resources, key=lambda e: float(np.linalg.norm(np.asarray(e.position) - np.asarray(player.position))))
        d = float(np.linalg.norm(np.asarray(nearest.position) - np.asarray(player.position)))
        if d < 0.5:  # standing on the resource cell: collect it
            self._collected = True
            self._done = True
            if self.controller.space.contains("collect"):
                return self._step_and_act(Action(name="collect", index=self.controller.space.index("collect")))
        return self._step_and_act(self._navigate(state, np.asarray(nearest.position), rng=rng))

    def should_terminate(self, state: WorldState) -> bool:
        return self._done or self._steps >= self.horizon

    def predict_outcome(self, state: WorldState) -> OutcomePrediction:
        resources = self._of_type(state, "resource")
        player = self._player(state)
        if not resources or player is None:
            return OutcomePrediction(self.name, expected_utility=0.0, expected_events=[], confidence=0.0)
        nearest = min(resources, key=lambda e: float(np.linalg.norm(np.asarray(e.position) - np.asarray(player.position))))
        d = float(np.linalg.norm(np.asarray(nearest.position) - np.asarray(player.position)))
        return OutcomePrediction(self.name, expected_utility=1.0 / (1.0 + d), expected_events=["move", "collect"], confidence=0.7)

    @property
    def collected(self) -> bool:
        return self._collected


class AvoidThreatSkill(RuleSkill):
    """Move away from the nearest visible enemy."""

    def __init__(self, controller: ActionController, horizon: int = 4, name: str = "avoid_threat", radius: float = 5.0) -> None:
        super().__init__(name=name, controller=controller, horizon=horizon)
        self.radius = float(radius)

    def can_start(self, state: WorldState) -> bool:
        return bool(self._of_type(state, "enemy"))

    def act(self, state: WorldState, rng=None) -> Action:
        enemies = self._of_type(state, "enemy")
        player = self._player(state)
        if not enemies or player is None:
            self._done = True
            return self._step_and_act(self.controller.from_index(0))
        nearest = min(enemies, key=lambda e: float(np.linalg.norm(np.asarray(e.position) - np.asarray(player.position))))
        d = float(np.linalg.norm(np.asarray(nearest.position) - np.asarray(player.position)))
        if d > self.radius:
            self._done = True
            return self._step_and_act(self.controller.from_index(0))
        # pick the move that maximises the distance to the nearest enemy
        p = np.asarray(player.position, dtype=np.float32)
        wall = self._grid_channel(state, 1)
        H, W = wall.shape
        best_delta = None
        best_dist = -1.0
        fallback: Optional[str] = None
        fallback_dist = -1.0
        for name, (dx, dy) in (("left", (-1, 0)), ("right", (1, 0)), ("up", (0, -1)), ("down", (0, 1))):
            gx, gy = int(round(p[0] + dx)), int(round(p[1] + dy))
            if not (0 <= gx < W and 0 <= gy < H) or wall[gy, gx] > 0.5:
                continue
            if not self.controller.space.contains(name):
                continue
            cand = p + np.array([dx, dy], dtype=np.float32)
            dist = min(float(np.linalg.norm(cand - np.asarray(e.position, dtype=np.float32))) for e in enemies)
            if fallback is None or dist > fallback_dist:
                fallback, fallback_dist = name, dist
            if dist > best_dist:
                best_dist = dist
                best_delta = name
        choice = best_delta if best_delta is not None else fallback
        if choice is not None:
            return self._step_and_act(Action(name=choice, index=self.controller.space.index(choice)))
        return self._step_and_act(self.controller.from_index(0))

    def predict_outcome(self, state: WorldState) -> OutcomePrediction:
        enemies = self._of_type(state, "enemy")
        player = self._player(state)
        if not enemies or player is None:
            return OutcomePrediction(self.name, expected_utility=0.0, expected_events=[], confidence=0.0)
        nearest = min(enemies, key=lambda e: float(np.linalg.norm(np.asarray(e.position) - np.asarray(player.position))))
        d = float(np.linalg.norm(np.asarray(nearest.position) - np.asarray(player.position)))
        return OutcomePrediction(self.name, expected_utility=min(d / self.radius, 1.0), expected_events=["move"], confidence=0.6)
