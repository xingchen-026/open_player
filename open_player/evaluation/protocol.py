"""Unified evaluation protocol (Phase 1.5).

Every experiment uses the same EvalProtocol: environment configuration,
episode budget, max steps and seed set.  MetricsCollector aggregates
per-run metric dicts into mean/std/median (never report a single run as a
conclusion).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EvalProtocol:
    """One uniform evaluation protocol shared by all agents/baselines."""

    name: str = "phase1_5"
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    episodes: int = 5
    max_steps: int = 100
    env_name: str = "world_a"
    extra: Dict[str, Any] = field(default_factory=dict)

    def seed_episodes(self) -> List[tuple]:
        return [(s, e) for s in self.seeds for e in range(self.episodes)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "seeds": list(self.seeds),
            "episodes": int(self.episodes),
            "max_steps": int(self.max_steps),
            "env_name": self.env_name,
            "extra": dict(self.extra),
        }


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """mean / std / median for every numeric key across rows."""
    if not rows:
        return {}
    numeric: Dict[str, List[float]] = {}
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric.setdefault(k, []).append(float(v))
    out: Dict[str, float] = {"n": float(len(rows))}
    for k, vals in numeric.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out[f"{k}_median"] = float(np.median(arr))
    return out


class MetricsCollector:
    """Collects rows then emits mean/std/median summaries."""

    def __init__(self, name: str = "metrics") -> None:
        self.name = name
        self.rows: List[Dict[str, Any]] = []

    def add(self, row: Dict[str, Any]) -> None:
        self.rows.append(dict(row))

    def summary(self) -> Dict[str, float]:
        return aggregate_rows(self.rows)

    def __len__(self) -> int:
        return len(self.rows)
