"""Training tests: replay, trainer online step, checkpoint roundtrip, loss down."""
from __future__ import annotations

import os

import numpy as np
import torch

from open_player.training.replay import ReplayBuffer
from open_player.training.trainer import WorldModelTrainer


def test_replay_buffer(cfg, schema, state_pair):
    rb = ReplayBuffer(capacity=8, seed=0)
    s0, s1 = state_pair
    for i in range(10):
        rb.store(s0, 3, s1, 0.1, False, 0.0)
    assert len(rb) == 8
    batch = rb.sample(4, schema, device="cpu")
    assert batch["state"].entities_t.shape == (4, schema.max_entities, 42)
    assert batch["action"].shape == (4,)
    assert batch["change"].shape == (4,)
    stats = rb.stats()
    assert stats["len"] == 8


def test_trainer_online_step(cfg, schema, state_pair, model):
    trainer = WorldModelTrainer(model, cfg, schema, device="cpu")
    s0, s1 = state_pair
    metrics = trainer.online_step(s0, 3, s1, 0.1, False, 0.0)
    assert "total" in metrics
    assert trainer.step == 1
    assert len(trainer.replay) == 1


def test_checkpoint_roundtrip(cfg, schema, state_pair, model, tmp_path):
    trainer = WorldModelTrainer(model, cfg, schema, device="cpu")
    s0, s1 = state_pair
    trainer.online_step(s0, 3, s1, 0.1, False, 0.0)
    path = str(tmp_path / "ckpt.pt")
    trainer.save_checkpoint(path, metrics={"total": 0.5})
    assert os.path.exists(path)
    # fresh model: load and compare
    from open_player.world.model import WorldModel
    model2 = WorldModel(schema, cfg, num_actions=6)
    trainer2 = WorldModelTrainer(model2, cfg, schema, device="cpu")
    meta = trainer2.load_checkpoint(path)
    assert meta["step"] == 1
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_loss_decreases_over_mini_loop(cfg):
    """End-to-end mini training: entity loss must decrease."""
    from open_player.agent.player import Player
    from open_player.environments.synthetic.env import SyntheticGridEnv
    cfg = cfg.merge({"training": {"steps": 120, "log_every": 1000, "replay_update_every": 1000, "update_every": 1}})
    env = SyntheticGridEnv(cfg)
    player = Player(cfg)
    # snapshot initial loss on the first transition
    obs0 = env.reset(seed=0)
    s0 = player.perceive(obs0, 0)
    obs1, _, _, _ = env.step(3)
    s1 = player.perceive(obs1, 1)
    pred = player.world_model.predict(s0, 3)
    with torch.no_grad():
        tz = player.world_model.representation(s1).z
    initial = float(player.world_model.loss(pred, s1, change_label=torch.tensor([0.0]), target_z=tz)["entity"])
    rep = player.learn(env, total_steps=120, verbose=False)
    final = rep.final_loss.get("entity", float("inf"))
    assert final < initial
    assert rep.events > 0
    assert len(player.trainer.replay) >= 100
