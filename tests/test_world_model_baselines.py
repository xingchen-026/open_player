"""Persistence / random-dynamics world model baselines."""
from __future__ import annotations

import torch

from open_player.core.config import load_config
from open_player.core.schema import SchemaSet
from open_player.environments.synthetic.env import SyntheticGridEnv
from open_player.evaluation.world_model_baselines import PersistenceWorldModel, RandomDynamicsWorldModel
from open_player.tracking.tracker import BeliefTracker
from open_player.world.model import WorldModel


def _sequence(length=17, seed=2):
    cfg = load_config("configs/phase1_5.yaml")
    cfg.device = "cpu"
    schema = SchemaSet.from_config(cfg)
    env = SyntheticGridEnv(cfg)
    tr = BeliefTracker(schema)
    obs = env.reset(seed=seed)
    states = [tr.track(None, obs, 0)]
    actions = []
    for i in range(length - 1):
        obs, _, _, _ = env.step((i * 3) % 6)
        states.append(tr.track(states[-1], obs, i + 1))
        actions.append((i * 3) % 6)
    return cfg, schema, states, actions


def test_persistence_errors_grow_with_horizon():
    cfg, schema, states, actions = _sequence()
    p = PersistenceWorldModel(schema)
    errs = p.prediction_errors(states[0], actions, states[1:], horizons=(1, 4, 8, 16))
    for key in ("step1_entity", "step4_entity", "step8_entity", "step16_entity", "step1_spatial", "step16_latent"):
        assert key in errs
    assert errs["step8_entity"] >= errs["step1_entity"] - 1e-9
    assert p.num_parameters() == 0


def test_random_dynamics_interface_matches():
    cfg, schema, states, actions = _sequence()
    r = RandomDynamicsWorldModel(schema, cfg, num_actions=6)
    assert r.num_parameters() > 0
    errs = r.prediction_errors(states[0], actions, states[1:], horizons=(1, 4, 8))
    for key in ("step1_entity", "step4_entity", "step8_entity", "step1_latent"):
        assert key in errs and torch.isfinite(torch.tensor(errs[key]))


def test_learned_model_present_for_comparison():
    cfg, schema, states, actions = _sequence()
    m = WorldModel(schema, cfg, num_actions=6)
    errs = m.prediction_errors(states[0], actions, states[1:], horizons=(1, 4, 8, 16))
    assert "step16_entity" in errs
