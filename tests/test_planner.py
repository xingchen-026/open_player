"""Planner tests: horizon tiers, candidate scoring, rollout utility."""
from __future__ import annotations

from open_player.actions.controller import ActionController
from open_player.actions.specs import DiscreteActionSpace
from open_player.core.types import Goal, GoalType
from open_player.planning.planner import Planner
from open_player.skills.registry import SkillRegistry


def test_horizons_from_config(cfg, schema):
    planner = Planner(cfg, SkillRegistry.build_default(ActionController(DiscreteActionSpace(["noop", "up", "down", "left", "right", "collect"])), cfg), schema, model=None)
    assert planner.horizon_for(Goal("g", GoalType.TASK.value)) == 4
    assert planner.horizon_for(Goal("g", GoalType.EXPLORATION.value)) == 8
    assert planner.horizon_for(Goal("g", GoalType.LEARNING.value)) == 32


def test_plan_selects_goal_skill(cfg, schema, state_pair):
    reg = SkillRegistry.build_default(ActionController(DiscreteActionSpace(["noop", "up", "down", "left", "right", "collect"])), cfg)
    planner = Planner(cfg, reg, schema, model=None)
    goal = Goal("g1", GoalType.TASK.value, target="resource")
    plan = planner.plan(state_pair[0], goal)
    assert plan.skill_name in ("collect", "approach_resource", "explore", "avoid_threat")
    assert plan.horizon == 4
    assert isinstance(plan.scores, dict) and plan.scores


def test_plan_with_world_model(cfg, schema, state_pair, model):
    reg = SkillRegistry.build_default(ActionController(DiscreteActionSpace(["noop", "up", "down", "left", "right", "collect"])), cfg)
    planner = Planner(cfg, reg, schema, model=model)
    goal = Goal("g2", GoalType.EXPLORATION.value, target="unknown")
    plan = planner.plan(state_pair[0], goal)
    assert plan.skill_name == "explore"
    assert plan.horizon == 8
    # rollout evaluation should produce a float
    from open_player.planning.rollout import WorldModelRollout
    r = WorldModelRollout(model, schema, horizon=4)
    preds = r.rollout(state_pair[0], [1] * 4)
    util = r.evaluate(state_pair[0], preds, goal)
    assert isinstance(util, float)
