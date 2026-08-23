"""Skill tests: initiation / act / termination / outcome model."""
from __future__ import annotations

import numpy as np

from open_player.actions.controller import ActionController
from open_player.actions.specs import DiscreteActionSpace
from open_player.skills.registry import SkillRegistry
from open_player.skills.rule import CollectSkill, ExploreSkill


def _controller():
    return ActionController(DiscreteActionSpace(["noop", "up", "down", "left", "right", "collect"]))


def test_explore_skill_acts(state_pair):
    skill = ExploreSkill(_controller())
    state = state_pair[0]
    assert skill.can_start(state)
    action = skill.act(state)
    assert action.index < 6
    outcome = skill.predict_outcome(state)
    assert outcome.skill_name == "explore"
    skill.reset()
    assert skill.steps == 0


def test_collect_skill_collects_when_on_resource(cfg, schema, tracker):
    from open_player.environments.synthetic.env import SyntheticGridEnv
    env = SyntheticGridEnv(cfg)
    env.reset(seed=3)
    w = env.world
    r = w.resources[0]
    r.position = w.player_pos.copy()
    obs = env.world.build_observation(t=0)
    state = tracker.track(None, obs, t=0)
    skill = CollectSkill(_controller())
    assert skill.can_start(state)
    action = skill.act(state)
    assert action.name == "collect"
    assert skill.should_terminate(state)


def test_skill_terminates_on_horizon(state_pair):
    skill = ExploreSkill(_controller(), horizon=2)
    state = state_pair[0]
    for _ in range(2):
        skill.act(state)
    assert skill.should_terminate(state)


def test_registry_default(cfg):
    reg = SkillRegistry.build_default(_controller(), cfg)
    assert {"explore", "collect", "approach_resource", "avoid_threat"} <= set(reg.names())
    assert len(reg) >= 4
