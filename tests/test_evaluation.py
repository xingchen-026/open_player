"""Evaluation framework: logger, baselines, world model eval, curves."""
from __future__ import annotations

import json
import os

import torch

from open_player.agent.player import Player
from open_player.evaluation.benchmark import evaluate_baseline, evaluate_world_model
from open_player.evaluation.logger import ExperimentLogger


def test_logger_csv_jsonl(tmp_path):
    logger = ExperimentLogger(str(tmp_path), name="test")
    logger.record(0, loss=0.5, extra={"a": 1})
    logger.record(10, loss=0.2, extra={"a": 2})
    logger.close()
    assert os.path.exists(str(tmp_path / "test.csv"))
    assert os.path.exists(str(tmp_path / "test.jsonl"))
    rows = ExperimentLogger.load_csv(str(tmp_path / "test.csv"))
    assert len(rows) == 2
    assert rows[1]["loss"] == 0.2
    assert rows[0]["step"] == 0


def test_baselines_run(cfg_p1, schema_p1):
    from open_player.environments.synthetic.env import SyntheticGridEnv
    env = SyntheticGridEnv(cfg_p1)
    from open_player.tracking.tracker import BeliefTracker
    tr = BeliefTracker(schema_p1)

    def perceive(obs, t):
        return tr.track(None, obs, t)

    for kind in ("random", "rule"):
        summary = evaluate_baseline(env, kind, perceive, episodes=2, max_steps=30, schema=schema_p1, seed=0)
        assert summary["kind"] == kind
        assert "mean_exploration_coverage" in summary
        assert "mean_collected" in summary


def test_evaluate_world_model(cfg_p1, schema_p1):
    from open_player.environments.synthetic.env import SyntheticGridEnv
    from open_player.world.model import WorldModel
    from open_player.tracking.tracker import BeliefTracker
    env = SyntheticGridEnv(cfg_p1)
    tr = BeliefTracker(schema_p1)
    model = WorldModel(schema_p1, cfg_p1, num_actions=6)

    def perceive(obs, t):
        return tr.track(None, obs, t)

    errs = evaluate_world_model(model, perceive, env, steps=8, seed=0)
    for key in ("step1_entity", "step4_entity", "step8_entity", "step4_latent", "step8_latent"):
        assert key in errs
    assert errs["step1_entity"] >= 0


def test_player_evaluate_and_skill_api(cfg_p1):
    from open_player.environments.transfer import make_transfer_envs
    pair = make_transfer_envs(cfg_p1)
    player = Player(cfg_p1)
    summary = player.evaluate(pair.world_a, episodes=2, max_steps=30)
    assert summary["episodes"] == 2
    assert "goal_success_rate" in summary
    assert "mean_exploration_coverage" in summary
    report = player.train_skill(pair.world_a, steps=120, verbose=False)
    assert report.action_accuracy > 1.0 / 6.0
    assert "neural_explore" in player.registry.names()


def test_phase1_checkpoint_roundtrip(cfg_p1, tmp_path):
    from open_player.environments.transfer import make_transfer_envs
    pair = make_transfer_envs(cfg_p1)
    player = Player(cfg_p1)
    player.learn(pair.world_a, total_steps=20, verbose=False)
    path = str(tmp_path / "phase1.pt")
    player.save_checkpoint(path)
    # fresh player loads model + vision + predictor
    player2 = Player(cfg_p1)
    meta = player2.load_checkpoint(path)
    assert meta["step"] == 20
    for p1, p2 in zip(player.world_model.parameters(), player2.world_model.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(player.vision.parameters(), player2.vision.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(player.world_model.change_predictor.parameters(), player2.world_model.change_predictor.parameters()):
        assert torch.allclose(p1, p2)


def test_evaluate_transfer_small(cfg_p1):
    from open_player.environments.transfer import make_transfer_envs
    cfg = cfg_p1.merge({"training": {"steps": 30}})
    pair = make_transfer_envs(cfg)
    player = Player(cfg)
    results = player.evaluate_transfer(
        train_env=pair.world_a,
        test_env=pair.world_b,
        steps=30,
        adaptation_steps=20,
        episodes=1,
        max_steps=20,
    )
    assert "baselines_b" in results
    assert "world_b" in results and "zero_shot" in results["world_b"]
    assert "after_adaptation" in results["world_b"]
    assert "prediction_errors_b_zero_shot" in results
