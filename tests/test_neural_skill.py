"""NeuralSkill: interface, BC training, goal completion."""
from __future__ import annotations

import numpy as np

from open_player.core.config import set_seed
from open_player.environments.synthetic.env import SyntheticGridEnv
from open_player.evaluation.baselines import RuleBaseline
from open_player.skills.neural import NeuralSkill, StateFeaturizer
from open_player.tracking.tracker import BeliefTracker
from open_player.training.skill_trainer import SkillTrainer


def _easy_cfg(cfg_p1):
    """No-enemy open world: rule teacher explores fully and collects."""
    return cfg_p1.merge({
        "environment": {
            "grid_size": 10,
            "num_enemies": 0,
            "num_resources": 1,
            "fog_radius": 20,
            "max_steps": 60,
            "player_hp": 10,
            "render_rgb": True,
        },
    })


def _train_skill(cfg, schema, env, steps=600, epochs=30):
    set_seed(int(cfg.seed))
    featurizer = StateFeaturizer(schema)
    trainer = SkillTrainer(cfg, featurizer, device="cpu")
    trainer.epochs = epochs
    tr = BeliefTracker(schema)
    policy = RuleBaseline(env.action_space, schema, seed=0)

    def perceive(obs, t):
        return tr.track(None, obs, t)

    xs, actions, terms, info = trainer.collect(env, perceive, policy.act, steps=steps, seed=0)
    skill = NeuralSkill("neural_explore", list(env.action_space.names), featurizer=featurizer).to("cpu")
    report = trainer.train(skill, xs, actions, terms)
    return skill, report, perceive


def test_neural_skill_interface(cfg_p1, schema_p1, env_p1):
    featurizer = StateFeaturizer(schema_p1)
    skill = NeuralSkill("neural_explore", ["noop", "up", "down", "left", "right", "collect"], featurizer=featurizer, horizon=4)
    assert skill.num_parameters() < 1_000_000
    tr = BeliefTracker(schema_p1)
    obs = env_p1.reset(seed=5)
    state = tr.track(None, obs, 0)
    assert skill.can_start(state)
    for _ in range(4):
        action = skill.act(state)
        assert 0 <= action.index < 6
    assert skill.should_terminate(state)
    outcome = skill.predict_outcome(state)
    assert outcome.skill_name == "neural_explore"
    skill.reset()
    assert skill.steps == 0


def test_bc_training_beats_random(cfg_p1, schema_p1):
    cfg = _easy_cfg(cfg_p1)
    env = SyntheticGridEnv(cfg)
    skill, report, _ = _train_skill(cfg, schema_p1, env, steps=300, epochs=15)
    assert report.action_accuracy > 1.0 / 6.0 + 0.1  # clearly above uniform


def test_neural_skill_completes_goal(cfg_p1, schema_p1):
    """The learned skill achieves its goal: exploration coverage >> random.

    (Its goal is exploration - it is the learned ExploreSkill replacement;
    'goal completion' for the full agent loop is covered by the Phase 0
    tests and the demos.)
    """
    cfg = _easy_cfg(cfg_p1)
    env = SyntheticGridEnv(cfg)
    skill, report, perceive = _train_skill(cfg, schema_p1, env, steps=600, epochs=30)
    assert report.action_accuracy > 0.8

    def coverage(seed):
        obs = env.reset(seed=seed)
        state = perceive(obs, 0)
        for t in range(50):
            action = skill.act(state)
            obs2, _r, done, _i = env.step(action)
            state = perceive(obs2, t + 1)
            if done:
                break
            if skill.should_terminate(state):
                skill.reset()
        w = env.world
        return len(w.visited) / max(w.grid_size * w.grid_size - len(w.walls), 1)

    covs = [coverage(11), coverage(12)]
    assert all(c > 0.5 for c in covs)
    assert sum(covs) / len(covs) > 0.7


def test_neural_skill_validity_masking(cfg_p1, schema_p1):
    """Blocked moves (walls) are masked: no pushing against geometry."""
    featurizer = StateFeaturizer(schema_p1)
    skill = NeuralSkill("n", ["noop", "up", "down", "left", "right", "collect"], featurizer=featurizer)
    tr = BeliefTracker(schema_p1)
    from open_player.environments.synthetic.env import SyntheticGridEnv
    env = SyntheticGridEnv(cfg_p1)
    obs = env.reset(seed=5)
    state = tr.track(None, obs, 0)
    mask = skill._valid_move_mask(state)
    assert mask.shape == (6,)
    player = [e for e in state.entity_states(0) if e.semantic_type == "player"][0]
    px, py = int(round(player.position[0])), int(round(player.position[1]))
    from open_player.core.state import grid_channel
    wall = grid_channel(state, 1)
    for i, name in enumerate(["noop", "up", "down", "left", "right", "collect"]):
        if name in ("noop", "collect"):
            assert bool(mask[i])
        elif name == "up" and wall[py - 1, px] > 0.5:
            assert not bool(mask[i])
