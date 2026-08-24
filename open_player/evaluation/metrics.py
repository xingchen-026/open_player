"""Episode / aggregate metrics for Phase 1 evaluation."""
from __future__ import annotations

from typing import Any, Dict, List


def episode_metrics(env: Any, info: Dict[str, Any], total_reward: float, steps: int) -> Dict[str, Any]:
    """Metrics of one finished environment episode."""
    w = env.world
    free = w.grid_size * w.grid_size - len(w.walls)
    coverage = len(w.visited) / max(free, 1)
    return {
        "steps": int(steps),
        "reward": float(total_reward),
        "collected": int(info.get("collected", 0)),
        "exploration_coverage": float(coverage),
        "death": bool(info.get("death", False)),
        "timeout": bool(info.get("timeout", False)),
        "goal_succeeded": bool(info.get("goal_succeeded", False)),
    }


def summarize_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-episode metrics into means (+ counts)."""
    if not entries:
        return {}
    numeric = ("reward", "collected", "exploration_coverage", "steps")
    boolean = ("death", "goal_succeeded")
    out: Dict[str, Any] = {"episodes": len(entries)}
    for key in numeric:
        vals = [float(e.get(key, 0.0)) for e in entries if key in e]
        if vals:
            out[f"mean_{key}"] = sum(vals) / len(vals)
    for key in boolean:
        vals = [bool(e.get(key, False)) for e in entries]
        out[f"rate_{key}"] = sum(vals) / len(vals)
    out["goal_success_rate"] = out.get("rate_goal_succeeded", 0.0)
    out["mean_exploration_coverage"] = out.get("mean_exploration_coverage", 0.0)
    out["mean_collected"] = out.get("mean_collected", 0.0)
    return out
