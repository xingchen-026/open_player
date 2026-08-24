"""Scheduled rollout (teacher forcing -> mixed -> model rollout) tests."""
from __future__ import annotations

import torch

from open_player.core.schema import SchemaSet
from open_player.environments.synthetic.env import SyntheticGridEnv
from open_player.tracking.tracker import BeliefTracker
from open_player.training.trainer import WorldModelTrainer
from open_player.world.model import WorldModel


def test_teacher_forcing_anneals(cfg_p1, schema_p1):
    model = WorldModel(schema_p1, cfg_p1, num_actions=6)
    trainer = WorldModelTrainer(model, cfg_p1, schema_p1, device="cpu")
    trainer.tf_initial = 0.9
    trainer.tf_final = 0.2
    trainer.tf_anneal_steps = 100
    assert trainer.teacher_forcing_ratio() == 0.9
    trainer.step = 50
    mid = trainer.teacher_forcing_ratio()
    trainer.step = 100
    end = trainer.teacher_forcing_ratio()
    assert 0.2 < mid < 0.9
    assert abs(end - 0.2) < 1e-6


def test_rollout_ratio_selects_model_rollout(cfg_p1, schema_p1):
    model = WorldModel(schema_p1, cfg_p1, num_actions=6)
    trainer = WorldModelTrainer(model, cfg_p1, schema_p1, device="cpu")
    trainer.rollout_ratio = 1.0
    trainer.tf_initial = trainer.tf_final = 1.0
    assert trainer._multi_step_teacher_forcing() == 0.0
    trainer.rollout_ratio = 0.0
    assert trainer._multi_step_teacher_forcing() == 1.0


def test_trainer_emits_ms_metrics(cfg_p1, schema_p1):
    set_seed = __import__("open_player.core.config", fromlist=["set_seed"]).set_seed
    set_seed(0)
    env = SyntheticGridEnv(cfg_p1)
    tr = BeliefTracker(schema_p1)
    model = WorldModel(schema_p1, cfg_p1, num_actions=6)
    trainer = WorldModelTrainer(model, cfg_p1, schema_p1, device="cpu")
    obs = env.reset(seed=1)
    state = tr.track(None, obs, 0)
    last = {}
    for i in range(9):
        obs2, _, _, _ = env.step(i % 6)
        state2 = tr.track(state, obs2, i + 1)
        m = trainer.online_step(state, i % 6, state2, 0.0, False, 0.0)
        if m:
            last = m
        state = state2
    assert "ms_step4" in last
    assert "ms_step8" in last
    assert "teacher_forcing" in last
