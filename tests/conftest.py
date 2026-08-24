"""Shared fixtures: small CPU config, schema, env, world states."""
from __future__ import annotations

import numpy as np
import pytest

from open_player.core.config import default_config, phase1_config, set_seed
from open_player.core.schema import SchemaSet
from open_player.environments.synthetic.env import SyntheticGridEnv
from open_player.tracking.tracker import BeliefTracker
from open_player.world.model import WorldModel


def make_small_config(**overrides: object) -> object:
    cfg = default_config().merge({
        "device": "cpu",
        "seed": 1234,
        "environment": {
            "grid_size": 10,
            "num_enemies": 1,
            "num_resources": 2,
            "fog_radius": 8,
            "max_steps": 40,
            "player_hp": 4,
        },
        "training": {
            "batch_size": 16,
            "replay_capacity": 256,
            "replay_update_every": 8,
            "steps": 200,
            "log_every": 1000,
        },
    })
    if overrides:
        cfg = cfg.merge(overrides)
    return cfg


def make_small_phase1_config(**overrides: object) -> object:
    """Phase 1 config sized for fast CPU tests."""
    cfg = phase1_config().merge({
        "device": "cpu",
        "seed": 7,
        "environment": {
            "grid_size": 10,
            "num_enemies": 1,
            "num_resources": 2,
            "fog_radius": 8,
            "max_steps": 40,
            "player_hp": 6,
            "render_rgb": True,
        },
        "training": {
            "batch_size": 16,
            "replay_capacity": 256,
            "replay_update_every": 1000,
            "steps": 150,
            "log_every": 1000,
        },
        "multi_step": {"horizons": [4, 8], "loss_weights": {"step1": 1.0, "step4": 0.5, "step8": 0.25}, "window": 8},
    })
    if overrides:
        cfg = cfg.merge(overrides)
    return cfg


@pytest.fixture(scope="session")
def cfg():
    return make_small_config()


@pytest.fixture(scope="session")
def schema(cfg):
    return SchemaSet.from_config(cfg)


@pytest.fixture()
def env(cfg):
    set_seed(int(cfg.seed))
    e = SyntheticGridEnv(cfg)
    e.reset(seed=42)
    return e


@pytest.fixture()
def tracker(schema):
    return BeliefTracker(schema, device="cpu")


@pytest.fixture()
def state_pair(env, tracker):
    obs0 = env.reset(seed=1)
    s0 = tracker.track(None, obs0, t=0)
    obs1, _r, _d, _i = env.step(3)
    s1 = tracker.track(s0, obs1, t=1)
    return s0, s1


@pytest.fixture(scope="session")
def model(cfg, schema):
    set_seed(0)
    m = WorldModel(schema, cfg, num_actions=6)
    return m


# ---------------- Phase 1 fixtures ---------------- #

@pytest.fixture(scope="session")
def cfg_p1():
    return make_small_phase1_config()


@pytest.fixture(scope="session")
def schema_p1(cfg_p1):
    return SchemaSet.from_config(cfg_p1)


@pytest.fixture()
def env_p1(cfg_p1):
    set_seed(int(cfg_p1.seed))
    e = SyntheticGridEnv(cfg_p1)
    e.reset(seed=5)
    return e
