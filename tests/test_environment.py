"""Synthetic environment tests."""
from __future__ import annotations

import numpy as np

from open_player.core.types import Action
from open_player.environments.synthetic.env import SyntheticGridEnv
from open_player.observation.dummy import DummyVisionEncoder


def test_reset_observation(env, schema):
    obs = env.reset(seed=42)
    types = [e.semantic_type for e in obs.entities]
    assert "player" in types
    assert "enemy" in types
    assert "resource" in types
    assert obs.spatial.shape == (8, 10, 10)
    assert obs.global_features.shape == (6,)
    # encoder -> world state
    ws = DummyVisionEncoder(schema).encode(obs)
    assert ws.entities_t.shape[1] == 32


def test_step_moves_player(env):
    env.reset(seed=5)
    before = env.world.player_pos.copy()
    env.step(Action("down", 3))
    after = env.world.player_pos
    moved = np.abs(before - after).sum() > 0
    # player only moves if the cell is free; retry a couple of steps
    for _ in range(5):
        if moved:
            break
        env.step(Action("down", 3))
        moved = np.abs(before - after).sum() > 0
        before = env.world.player_pos.copy()
    assert moved


def test_collect_reward(env):
    env.reset(seed=9)
    w = env.world
    r = w.resources[0]
    r.position = w.player_pos.copy()
    r.collected = False
    obs, reward, done, info = env.step(Action("collect", 5))
    assert info["collected_this_step"] is True
    assert reward >= 1.0


def test_deterministic_reset(env):
    o1 = env.reset(seed=11)
    o2 = env.reset(seed=11)
    p1 = [(e.entity_id, e.position.tolist()) for e in o1.entities]
    p2 = [(e.entity_id, e.position.tolist()) for e in o2.entities]
    assert p1 == p2


def test_death_ends_episode(env):
    env.reset(seed=13)
    w = env.world
    w.player_hp = 1
    e = w.enemies[0]
    e.position = w.player_pos + np.array([0.0, 1.0])
    obs, reward, done, info = env.step(0)
    assert done
    assert info["hp"] <= 0
    assert reward < 0


def test_action_space(cfg):
    env = SyntheticGridEnv(cfg)
    assert env.action_space.n == 6
    assert env.action_space.index("collect") == 5
