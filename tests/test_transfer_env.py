"""Transfer worlds: structural difference + vectorized environment."""
from __future__ import annotations

import numpy as np

from open_player.environments.synthetic.vector_env import SyntheticGridVecEnv
from open_player.environments.transfer import make_transfer_envs, world_structural_summary


def test_worlds_are_structurally_different(cfg_p1):
    pair = make_transfer_envs(cfg_p1)
    a = world_structural_summary(pair.world_a)
    b = world_structural_summary(pair.world_b)
    assert a["grid_size"] != b["grid_size"]
    assert b["interior_walls"] > a["interior_walls"]
    assert b["enemy_move_prob"] > a["enemy_move_prob"]
    assert a["resource_cluster"] is True and b["resource_cluster"] is False
    assert b["resource_spread"] > a["resource_spread"]


def test_transfer_env_deterministic(cfg_p1):
    pair1 = make_transfer_envs(cfg_p1)
    pair2 = make_transfer_envs(cfg_p1)
    o1 = pair1.world_b.reset(seed=9)
    o2 = pair2.world_b.reset(seed=9)
    p1 = [(e.entity_id, e.position.tolist()) for e in o1.entities]
    p2 = [(e.entity_id, e.position.tolist()) for e in o2.entities]
    assert p1 == p2
    assert pair1.world_b.extra if hasattr(pair1.world_b, "extra") else True


def test_vectorized_env(cfg_p1):
    vec = SyntheticGridVecEnv(cfg_p1, n_envs=4, seed=0)
    obs = vec.reset()
    assert len(obs) == 4
    assert all("rgb" in o.extra for o in obs)
    obs2, rewards, dones, infos = vec.step([1, 2, 3, 4])
    assert len(obs2) == 4
    assert rewards.shape == (4,)
    assert dones.shape == (4,)
    assert len(infos) == 4
    assert vec.action_space.n == 6


def test_vec_env_deterministic(cfg_p1):
    v1 = SyntheticGridVecEnv(cfg_p1, n_envs=2, seed=3)
    v2 = SyntheticGridVecEnv(cfg_p1, n_envs=2, seed=3)
    o1 = v1.reset()
    o2 = v2.reset()
    for a, b in zip(o1, o2):
        pa = [(e.entity_id, e.position.tolist()) for e in a.entities]
        pb = [(e.entity_id, e.position.tolist()) for e in b.entities]
        assert pa == pb
