"""ASCII renderer for the synthetic grid world (debugging / demos)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from open_player.core.types import EntityState, Observation

_LEGEND = {"P": "player", "E": "enemy", "R": "resource", "#": "wall", ".": "unknown", " ": "visited"}


class AsciiRenderer:
    """Renders observations or whole environments as text grids."""

    def render_observation(self, obs: Observation) -> str:
        g = obs.extra.get("grid_size", obs.spatial.shape[1])
        walls = set(obs.extra.get("walls", []))
        spatial = obs.spatial
        raw = list(obs.extra.get("raw_channels", []))
        def ch(name: str) -> np.ndarray:
            if name in raw:
                return spatial[raw.index(name)]
            return np.zeros((g, g), dtype=np.float32)
        wall = ch("wall")
        visited = ch("visited")
        player = ch("player")
        enemy = ch("enemy")
        resource = ch("resource")
        lines: List[str] = []
        for y in range(g):
            row = []
            for x in range(g):
                if player[y, x] > 0:
                    row.append("P")
                elif enemy[y, x] > 0:
                    row.append("E")
                elif resource[y, x] > 0:
                    row.append("R")
                elif wall[y, x] > 0:
                    row.append("#")
                elif visited[y, x] > 0:
                    row.append(" ")
                else:
                    row.append(".")
            lines.append("".join(row))
        lines.append("legend: " + " ".join(f"{k}={v}" for k, v in _LEGEND.items()))
        return "\n".join(lines)

    def render_env(self, env: Any) -> str:
        """Render the live environment (uses GridWorld.state_dict)."""
        sd: Dict[str, Any] = env.world.state_dict()
        g: int = sd["grid_size"]
        grid: Dict = {}
        for (x, y) in sd["walls"]:
            grid[(x, y)] = "#"
        for (x, y) in sd["visited"]:
            grid.setdefault((x, y), " ")
        for _, pos, collected in sd["resources"]:
            if not collected:
                grid[(int(pos[0]), int(pos[1]))] = "R"
        for _, pos, alive in sd["enemies"]:
            if alive:
                grid[(int(pos[0]), int(pos[1]))] = "E"
        px, py = sd["player_pos"]
        grid[(int(px), int(py))] = "P"
        lines: List[str] = []
        for y in range(g):
            row = []
            for x in range(g):
                row.append(grid.get((x, y), "."))
            lines.append("".join(row))
        lines.append(f"step={sd['steps']} hp={sd['player_hp']} collected={sd['collected']}")
        lines.append("legend: " + " ".join(f"{k}={v}" for k, v in _LEGEND.items()))
        return "\n".join(lines)
