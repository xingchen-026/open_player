"""STRICT_RGB_MODE leakage tests: no GT reaches decision modules."""
from __future__ import annotations

import numpy as np
import torch

from open_player.agent.player import Player
from open_player.core.config import load_config
from open_player.core.state import structured_grid
from open_player.environments.synthetic.env import SyntheticGridEnv
from open_player.observation.vision import LearnedVisionEncoder


def _strict_cfg(tmp_path=None):
    cfg = load_config("configs/phase1_5.yaml")
    cfg.device = "cpu"
    cfg.seed = 3
    cfg.vision.mode = "strict"
    cfg.training.log_every = 1000
    return cfg


def test_strict_mode_encoder_hides_positions():
    cfg = load_config("configs/phase1_5.yaml")
    cfg.device = "cpu"
    cfg.vision.mode = "strict"
    env = SyntheticGridEnv(cfg)
    obs = env.reset(seed=4)
    from open_player.core.schema import SchemaSet
    schema = SchemaSet.from_config(cfg)
    enc = LearnedVisionEncoder(schema, cfg, device="cpu")
    state = enc.encode(obs, t=0)
    assert state.metadata["strict_rgb"] is True
    p_s, p_e = schema.entity.field_slice("position")
    v_s, v_e = schema.entity.field_slice("velocity")
    # no GT position for non-player entities; player position comes from the
    # learned localisation head (not the observation)
    gt_player = np.asarray(next(e.position for e in obs.entities if e.semantic_type == "player"))
    dec = state.entity_states(0)
    for e, vec in zip(dec, state.entities_t[0].detach().cpu().numpy()):
        if e.semantic_type == "player":
            learned_xy = vec[p_s:p_e] * schema.entity.world_size
            assert not np.allclose(learned_xy, gt_player, atol=1e-3)
        elif e.semantic_type != "empty":
            assert np.allclose(vec[p_s:p_e], 0.0)
    assert np.allclose(state.relations_t.numpy(), 0.0)


def test_strict_mode_structured_grid_uses_learned_estimates():
    cfg = load_config("configs/phase1_5.yaml")
    cfg.device = "cpu"
    cfg.vision.mode = "learned_grid"
    env = SyntheticGridEnv(cfg)
    obs = env.reset(seed=4)
    from open_player.core.schema import SchemaSet
    schema = SchemaSet.from_config(cfg)
    enc = LearnedVisionEncoder(schema, cfg, device="cpu")
    state = enc.encode(obs, t=0)
    # GT struct_grid must NOT be stored in learned_grid / strict modes
    assert not state.metadata.get("struct_grid")
    wall = structured_grid(state, "wall")
    raw = list(obs.extra["raw_channels"])
    gt_wall = obs.spatial[raw.index("wall")]
    assert wall.shape == gt_wall.shape
    # the learned estimate differs from GT at init (it is an estimate)
    assert not np.allclose(wall, gt_wall)
    # threat is unavailable in learned modes -> zeros
    assert np.allclose(structured_grid(state, "threat"), 0.0)


def test_strict_player_env_info_sanitized():
    cfg = _strict_cfg()
    player = Player(cfg)
    dirty = {"hp": 6, "hp_delta": -1, "collected": 3, "collected_this_step": True,
             "threat_level": 0.7, "action": 3, "env_t": 5, "player_pos": [1, 1]}
    clean = player._sanitize_env_info(dirty)
    assert "hp" not in clean and "collected" not in clean
    assert "threat_level" not in clean and "player_pos" not in clean
    assert clean["action"] == 3
    assert clean["strict_rgb"] is True


def test_strict_player_uses_learned_event_emitter():
    cfg = _strict_cfg()
    player = Player(cfg)
    from open_player.events.detector import LearnedEventEmitter
    assert isinstance(player.detector, LearnedEventEmitter)
    assert player.vision_mode == "strict"


def test_strict_registry_excludes_rule_skills():
    cfg = _strict_cfg()
    player = Player(cfg)
    from open_player.environments.synthetic.env import SyntheticGridEnv
    env = SyntheticGridEnv(cfg)
    player.attach(env)
    names = set(player.registry.names())
    assert "collect" not in names
    assert "approach_resource" not in names
    assert "avoid_threat" not in names
    assert len(names) >= 1  # explore (or neural skill) over learned grids


def test_strict_goal_candidates_are_learned_only():
    from open_player.motivation.goals import GoalManager
    from open_player.motivation.motivation import IntrinsicMotivation
    cfg = load_config("configs/phase1_5.yaml")
    cfg.device = "cpu"
    cfg.vision.mode = "strict"
    env = SyntheticGridEnv(cfg)
    obs = env.reset(seed=4)
    from open_player.core.schema import SchemaSet
    schema = SchemaSet.from_config(cfg)
    enc = LearnedVisionEncoder(schema, cfg, device="cpu")
    state = enc.encode(obs, t=0)
    gm = GoalManager(cfg)
    mot = IntrinsicMotivation(cfg).compute(state)
    cands = gm.generate_candidates(state, mot, {"strict_rgb": True, "threat_level": 0.9})
    types = {g.goal_type for g in cands}
    assert "task" not in types and "survival" not in types
    assert types <= {"exploration", "learning", "information"}


def test_strict_player_smoke_run():
    """A strict-mode player can learn + evaluate without touching GT."""
    cfg = _strict_cfg()
    cfg.training.steps = 20
    from open_player.environments.synthetic.env import SyntheticGridEnv
    env = SyntheticGridEnv(cfg)
    player = Player(cfg)
    rep = player.learn(env, total_steps=20, verbose=False)
    assert rep.total_steps == 20
    summary = player.evaluate(env, episodes=1, max_steps=20)
    assert summary["episodes"] == 1
