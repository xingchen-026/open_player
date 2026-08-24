"""Multi-step world model training tests."""
from __future__ import annotations

import torch

from open_player.core.config import set_seed
from open_player.core.schema import SchemaSet
from open_player.environments.synthetic.env import SyntheticGridEnv
from open_player.tracking.tracker import BeliefTracker
from open_player.world.model import WorldModel


def _sequence(cfg_p1, schema_p1, length=9, seed=1):
    set_seed(int(cfg_p1.seed))
    env = SyntheticGridEnv(cfg_p1)
    tr = BeliefTracker(schema_p1)
    obs = env.reset(seed=seed)
    states = [tr.track(None, obs, 0)]
    actions = []
    for i in range(length - 1):
        obs, _, _, _ = env.step(i % 6)
        states.append(tr.track(states[-1], obs, i + 1))
        actions.append(i % 6)
    return states, actions


def test_multi_step_loss_keys(cfg_p1, schema_p1):
    model = WorldModel(schema_p1, cfg_p1, num_actions=6)
    states, actions = _sequence(cfg_p1, schema_p1)
    ms = model.multi_step_loss(states[0], actions, states[1:], teacher_forcing=1.0)
    assert "step4" in ms and "step8" in ms and "total_ms" in ms
    for k, v in ms.items():
        assert torch.isfinite(v)
    # pure model rollout is also finite
    ms_r = model.multi_step_loss(states[0], actions, states[1:], teacher_forcing=0.0)
    assert torch.isfinite(ms_r["total_ms"])


def test_prediction_errors_measurable(cfg_p1, schema_p1):
    model = WorldModel(schema_p1, cfg_p1, num_actions=6)
    states, actions = _sequence(cfg_p1, schema_p1)
    errs = model.prediction_errors(states[0], actions, states[1:])
    for key in ("step1_entity", "step4_entity", "step8_entity", "step1_spatial", "step4_latent", "step8_latent", "step4_entity_tf"):
        assert key in errs
    # teacher forcing is an upper bound: it should not be worse than rollout
    assert errs["step4_entity_tf"] <= errs["step4_entity"] + 0.05


def test_multi_step_loss_decreases(cfg_p1, schema_p1):
    """A few online updates reduce the multi-step total on a fixed sequence."""
    from open_player.training.trainer import WorldModelTrainer
    model = WorldModel(schema_p1, cfg_p1, num_actions=6)
    trainer = WorldModelTrainer(model, cfg_p1, schema_p1, device="cpu")
    states, actions = _sequence(cfg_p1, schema_p1)
    ms0 = float(model.multi_step_loss(states[0], actions, states[1:], teacher_forcing=1.0)["total_ms"])
    for i in range(len(states) - 1):
        trainer.online_step(states[i], actions[i], states[i + 1], 0.0, False, 0.0)
    ms1 = float(model.multi_step_loss(states[0], actions, states[1:], teacher_forcing=1.0)["total_ms"])
    assert ms1 < ms0
