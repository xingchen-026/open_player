"""Transfer worlds: structurally different synthetic world configurations.

World A (open map, clustered resources, slow enemies) is the training world.
World B (narrow corridors, scattered resources, fast enemies) and World C
(maze topology, edge-ring resources, medium enemies) are held-out test
worlds.  They differ structurally, not just by seed.

Phase 1.5 adds make_env_variant() for held-out generalization groups
(enemy speed interpolation/extrapolation, resource density).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from open_player.environments.synthetic.env import SyntheticGridEnv


@dataclass
class TransferPair:
    """World A (train) + held-out Worlds B / C."""

    world_a: SyntheticGridEnv
    world_b: SyntheticGridEnv
    world_c: Optional[SyntheticGridEnv] = None


def make_transfer_envs(config: Any) -> TransferPair:
    """Build World A / World B / World C from the config."""
    env_a = _env_for(config, config.get("transfer.world_a", {}))
    env_b = _env_for(config, config.get("transfer.world_b", {}))
    wc_cfg = config.get("validation.world_c", None) or config.get("transfer.world_c", None)
    env_c = _env_for(config, wc_cfg) if wc_cfg else None
    return TransferPair(world_a=env_a, world_b=env_b, world_c=env_c)


def make_env_variant(config: Any, overrides: Dict[str, Any]) -> SyntheticGridEnv:
    """World A with structural overrides (held-out generalization groups)."""
    world_cfg = dict(config.get("transfer.world_a", {}))
    world_cfg.update(overrides)
    return _env_for(config, world_cfg)


def _env_for(config: Any, world_cfg: Dict[str, Any]) -> SyntheticGridEnv:
    base = config.to_dict()
    env = dict(base.get("environment", {}))
    env.update(world_cfg)
    base["environment"] = env
    base["seed"] = int(config.seed)
    from open_player.core.config import Config
    return SyntheticGridEnv(Config(base))


def world_structural_summary(env: SyntheticGridEnv) -> Dict[str, Any]:
    """Compact structural fingerprint of a world (for reports/tests)."""
    w = env.world
    interior_walls = sum(1 for (x, y) in w.walls if 0 < x < w.grid_size - 1 and 0 < y < w.grid_size - 1)
    res_positions = np_positions(w.resources)
    centroid = res_positions.mean(axis=0) if len(res_positions) else None
    spread = float(res_positions.std(axis=0).sum()) if len(res_positions) > 1 else 0.0
    return {
        "grid_size": w.grid_size,
        "interior_walls": interior_walls,
        "num_enemies": w.num_enemies,
        "num_resources": w.num_resources,
        "enemy_move_prob": w.enemy_move_prob,
        "enemy_attack_prob": w.enemy_attack_prob,
        "fog_radius": w.fog_radius,
        "resource_cluster": w.resource_cluster,
        "resource_edge": w.resource_edge,
        "resource_centroid": None if centroid is None else centroid.tolist(),
        "resource_spread": spread,
    }


def np_positions(items: Any) -> Any:
    import numpy as np
    return np.stack([np.asarray(r.position, dtype=np.float32) for r in items]) if items else np.zeros((0, 2), dtype=np.float32)
