"""GridWorld: the simulation core of the Phase 0 synthetic environment.

Contains: player, enemies, resources, walls, unknown area (fog).  Behaviors:
move, approach, collision, collect, threat, death, explore.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from open_player.core.types import EntityState, Observation

# Raw spatial observation channels (order matters, exported for tests).
RAW_CHANNELS: Tuple[str, ...] = (
    "occupancy", "wall", "resource", "enemy", "player", "visited", "unknown", "threat",
)

# Global observation features.
GLOBAL_FEATURES: Tuple[str, ...] = ("hp_frac", "collected_frac", "step_frac", "threat_level", "novelty", "energy")

_DIRECTIONS: Dict[str, Tuple[int, int]] = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


@dataclass
class _Enemy:
    entity_id: str
    position: np.ndarray
    alive: bool = True


@dataclass
class _Resource:
    entity_id: str
    position: np.ndarray
    collected: bool = False


class GridWorld:
    """Deterministic-per-seed grid world simulation."""

    def __init__(
        self,
        grid_size: int = 12,
        num_enemies: int = 2,
        num_resources: int = 4,
        fog_radius: int = 5,
        player_hp: int = 3,
        enemy_move_prob: float = 0.8,
        enemy_attack_prob: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.grid_size = int(grid_size)
        self.num_enemies = int(num_enemies)
        self.num_resources = int(num_resources)
        self.fog_radius = int(fog_radius)
        self.player_hp_max = int(player_hp)
        self.enemy_move_prob = float(enemy_move_prob)
        self.enemy_attack_prob = float(enemy_attack_prob)
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        g = self.grid_size
        self.steps = 0
        self.player_hp = self.player_hp_max
        self.collected = 0
        self.hp_delta = 0
        self.collected_this_step = False
        self.explored_this_step = False

        # Walls: full border plus a few deterministic interior blocks.
        self.walls: Set[Tuple[int, int]] = set()
        for x in range(g):
            self.walls.add((x, 0))
            self.walls.add((x, g - 1))
        for y in range(g):
            self.walls.add((0, y))
            self.walls.add((g - 1, y))
        n_blocks = max(1, g // 4)
        for _ in range(n_blocks):
            bx = int(self.rng.integers(2, g - 4))
            by = int(self.rng.integers(2, g - 4))
            for dx in range(2):
                for dy in range(1):
                    self.walls.add((bx + dx, by + dy))

        def free_cell() -> np.ndarray:
            while True:
                p = self.rng.integers(1, g - 1, size=2)
                if tuple(p.tolist()) not in self.walls:
                    return p.astype(np.float32)

        self.player_pos = free_cell()
        self.enemies: List[_Enemy] = []
        for i in range(self.num_enemies):
            pos = free_cell()
            while np.abs(pos - self.player_pos).sum() < 3:
                pos = free_cell()
            self.enemies.append(_Enemy(entity_id=f"enemy-{i}", position=pos))
        self.resources: List[_Resource] = []
        for i in range(self.num_resources):
            pos = free_cell()
            while np.abs(pos - self.player_pos).sum() < 2:
                pos = free_cell()
            self.resources.append(_Resource(entity_id=f"resource-{i}", position=pos))

        self.visited: Set[Tuple[int, int]] = {(int(self.player_pos[0]), int(self.player_pos[1]))}

    # ------------------------------------------------------------------ #
    def step(self, action_name: str) -> Tuple[float, bool, Dict[str, Any]]:
        """Apply one action; returns (reward, done, info)."""
        self.steps += 1
        self.hp_delta = 0
        self.collected_this_step = False
        self.explored_this_step = False
        reward = 0.0
        info: Dict[str, Any] = {}

        # 1) player move
        if action_name in _DIRECTIONS:
            dx, dy = _DIRECTIONS[action_name]
            np_ = self.player_pos + np.array([dx, dy], dtype=np.float32)
            cell = (int(np_[0]), int(np_[1]))
            if cell not in self.walls and 0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size:
                blocked = any(e.alive and (int(e.position[0]), int(e.position[1])) == cell for e in self.enemies)
                if not blocked:
                    self.player_pos = np_
        elif action_name == "collect":
            for r in self.resources:
                if not r.collected and (int(r.position[0]), int(r.position[1])) == (int(self.player_pos[0]), int(self.player_pos[1])):
                    r.collected = True
                    self.collected += 1
                    self.collected_this_step = True
                    reward += 1.0
                    info["collected_entity"] = r.entity_id
                    break

        # 2) exploration
        for cell in self._neighborhood(self.player_pos, radius=1, include_self=True):
            if cell not in self.walls:
                if cell not in self.visited:
                    self.visited.add(cell)
                    self.explored_this_step = True
        if self.explored_this_step:
            reward += 0.02

        # 3) enemies approach / wander
        for e in self.enemies:
            if not e.alive:
                continue
            if self.rng.random() < self.enemy_move_prob and self._visible(e.position):
                e.position = self._step_towards(e.position, self.player_pos)
            else:
                e.position = self._random_step(e.position)

        # 4) damage & death (attacks are probabilistic)
        threat = self.threat_level()
        for e in self.enemies:
            if e.alive and self._adjacent(e.position, self.player_pos) and self.rng.random() < self.enemy_attack_prob:
                self.player_hp -= 1
                self.hp_delta -= 1
                info["damage_from"] = e.entity_id
                break
        dead = self.player_hp <= 0
        if dead:
            reward += -1.0
            info["death"] = True
        info.update({
            "hp": self.player_hp,
            "hp_max": self.player_hp_max,
            "hp_delta": self.hp_delta,
            "collected": self.collected,
            "collected_this_step": self.collected_this_step,
            "explored_this_step": self.explored_this_step,
            "threat_level": threat,
            "player_pos": self.player_pos.copy(),
            "player_hp": self.player_hp,
        })
        timeout = self.steps >= self._max_steps_hint
        done = dead or timeout
        return float(reward), bool(done), info

    # -- observation ---------------------------------------------------- #
    def build_observation(self, t: int = 0) -> Observation:
        g = self.grid_size
        spatial = np.zeros((len(RAW_CHANNELS), g, g), dtype=np.float32)
        wall_idx = RAW_CHANNELS.index("wall")
        visited_idx = RAW_CHANNELS.index("visited")
        unknown_idx = RAW_CHANNELS.index("unknown")
        occ_idx = RAW_CHANNELS.index("occupancy")
        for (x, y) in self.walls:
            spatial[wall_idx, y, x] = 1.0
        for (x, y) in self.visited:
            spatial[visited_idx, y, x] = 1.0
        # unknown = inside-grid cells never visited and not wall
        for y in range(g):
            for x in range(g):
                if (x, y) not in self.walls and (x, y) not in self.visited:
                    spatial[unknown_idx, y, x] = 1.0

        entities: List[EntityState] = []
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        spatial[RAW_CHANNELS.index("player"), py, px] = 1.0
        spatial[occ_idx, py, px] = 1.0
        entities.append(self._player_entity())

        for e in self.enemies:
            if not e.alive:
                continue
            if not self._visible(e.position):
                continue
            ex, ey = int(e.position[0]), int(e.position[1])
            spatial[RAW_CHANNELS.index("enemy"), ey, ex] = 1.0
            spatial[occ_idx, ey, ex] = 1.0
            entities.append(self._enemy_entity(e))
        for r in self.resources:
            if r.collected:
                continue
            if not self._visible(r.position):
                continue
            rx, ry = int(r.position[0]), int(r.position[1])
            spatial[RAW_CHANNELS.index("resource"), ry, rx] = 1.0
            spatial[occ_idx, ry, rx] = 1.0
            entities.append(self._resource_entity(r))

        # threat channel: distance falloff from alive enemies
        for y in range(g):
            for x in range(g):
                cell = np.array([x, y], dtype=np.float32)
                best = 0.0
                for e in self.enemies:
                    if e.alive:
                        d = float(np.abs(e.position - cell).max())
                        best = max(best, max(0.0, 1.0 - d / max(self.fog_radius, 1)))
                spatial[RAW_CHANNELS.index("threat"), y, x] = best

        global_features = np.array([
            self.player_hp / max(self.player_hp_max, 1),
            self.collected / max(self.num_resources, 1),
            min(self.steps / 1000.0, 1.0),
            self.threat_level(),
            1.0 - len(self.visited) / max(g * g, 1),
            1.0,
        ], dtype=np.float32)

        return Observation(
            entities=entities,
            spatial=spatial,
            global_features=global_features,
            t=t,
            extra={
                "walls": sorted(self.walls),
                "grid_size": g,
                "fog_radius": self.fog_radius,
                "raw_channels": list(RAW_CHANNELS),
                "global_features": list(GLOBAL_FEATURES),
            },
        )

    # -- entities ------------------------------------------------------- #
    def _player_entity(self) -> EntityState:
        dyn = np.zeros(12, dtype=np.float32)
        dyn[0] = self.player_hp / max(self.player_hp_max, 1)
        dyn[1] = 1.0
        return EntityState(
            entity_id="player",
            semantic_type="player",
            position=self.player_pos.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            size=1.0,
            status=self.player_hp / max(self.player_hp_max, 1),
            dynamics_features=dyn,
        )

    def _enemy_entity(self, e: _Enemy) -> EntityState:
        dyn = np.zeros(12, dtype=np.float32)
        dyn[0] = 1.0
        dyn[2] = 1.0
        return EntityState(
            entity_id=e.entity_id,
            semantic_type="enemy",
            position=e.position.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            size=1.0,
            status=1.0,
            dynamics_features=dyn,
        )

    def _resource_entity(self, r: _Resource) -> EntityState:
        dyn = np.zeros(12, dtype=np.float32)
        dyn[3] = 1.0
        return EntityState(
            entity_id=r.entity_id,
            semantic_type="resource",
            position=r.position.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            size=1.0,
            status=1.0,
            dynamics_features=dyn,
        )

    # -- helpers -------------------------------------------------------- #
    @property
    def _max_steps_hint(self) -> int:
        return 10 ** 9  # overridden by the env layer (no hidden state here)

    def _visible(self, pos: np.ndarray) -> bool:
        return float(np.abs(pos - self.player_pos).max()) <= self.fog_radius

    def _adjacent(self, a: np.ndarray, b: np.ndarray) -> bool:
        return float(np.abs(a - b).sum()) <= 1.0

    def _neighborhood(self, pos: np.ndarray, radius: int, include_self: bool) -> List[Tuple[int, int]]:
        out = []
        x, y = int(pos[0]), int(pos[1])
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if not include_self and dx == 0 and dy == 0:
                    continue
                cx, cy = x + dx, y + dy
                if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                    out.append((cx, cy))
        return out

    def _free(self, pos: np.ndarray, block_player: bool = False) -> bool:
        cell = (int(pos[0]), int(pos[1]))
        if cell in self.walls:
            return False
        if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
            return False
        if block_player and cell == (int(self.player_pos[0]), int(self.player_pos[1])):
            return False
        return not any(e.alive and (int(e.position[0]), int(e.position[1])) == cell for e in self.enemies)

    def _random_step(self, pos: np.ndarray) -> np.ndarray:
        dirs = list(_DIRECTIONS.values())
        self.rng.shuffle(dirs)
        for dx, dy in dirs:
            nxt = pos + np.array([dx, dy], dtype=np.float32)
            if self._free(nxt, block_player=True):
                return nxt
        return pos

    def _step_towards(self, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - pos
        options: List[Tuple[int, int]] = []
        ax, ay = abs(float(delta[0])), abs(float(delta[1]))
        if ax >= ay:
            options.append((int(np.sign(delta[0])), 0))
        if ay >= ax:
            options.append((0, int(np.sign(delta[1]))))
        self.rng.shuffle(options)
        for dx, dy in options:
            nxt = pos + np.array([dx, dy], dtype=np.float32)
            if self._free(nxt, block_player=True):
                return nxt
        return pos

    def threat_level(self) -> float:
        best = 0.0
        for e in self.enemies:
            if e.alive:
                d = float(np.abs(e.position - self.player_pos).max())
                best = max(best, max(0.0, 1.0 - d / max(self.fog_radius, 1)))
        return best

    def state_dict(self) -> Dict[str, Any]:
        """Serialisable snapshot (used by renderer and tests)."""
        return {
            "grid_size": self.grid_size,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "player_hp": self.player_hp,
            "collected": self.collected,
            "enemies": [(e.entity_id, e.position.tolist(), e.alive) for e in self.enemies],
            "resources": [(r.entity_id, r.position.tolist(), r.collected) for r in self.resources],
            "walls": sorted(self.walls),
            "visited": sorted(self.visited),
        }
