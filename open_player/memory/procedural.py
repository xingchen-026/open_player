"""ProceduralMemory: skill-level statistics (Phase 0: simple counters)."""
from __future__ import annotations

from typing import Any, Dict


class ProceduralMemory:
    """Records skill outcomes; the planner uses success rates."""

    def __init__(self) -> None:
        self._stats: Dict[str, Dict[str, float]] = {}

    def record(self, skill_name: str, success: bool, reward: float = 0.0) -> None:
        s = self._stats.setdefault(skill_name, {"attempts": 0.0, "successes": 0.0, "total_reward": 0.0})
        s["attempts"] += 1.0
        s["successes"] += float(bool(success))
        s["total_reward"] += float(reward)

    def success_rate(self, skill_name: str, default: float = 0.5) -> float:
        s = self._stats.get(skill_name)
        if not s or s["attempts"] <= 0:
            return float(default)
        return float(s["successes"] / s["attempts"])

    def stats(self, skill_name: Optional[str] = None) -> Dict[str, Any]:
        if skill_name is not None:
            return dict(self._stats.get(skill_name, {}))
        return {k: dict(v) for k, v in self._stats.items()}
