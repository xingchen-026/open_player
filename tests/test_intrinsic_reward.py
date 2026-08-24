"""Intrinsic reward: components, penalties, goal integration."""
from __future__ import annotations

import numpy as np

from open_player.motivation.intrinsic import IntrinsicReward, VisitCounter
from open_player.tracking.tracker import BeliefTracker


def test_components_and_penalties(cfg_p1, schema_p1, env_p1):
    ir = IntrinsicReward(cfg_p1)
    vc = VisitCounter(cap=10)
    tr = BeliefTracker(schema_p1)
    obs = env_p1.reset(seed=5)
    state = tr.track(None, obs, 0)
    player = [e for e in state.entity_states(0) if e.semantic_type == "player"][0]
    base = {"state": state, "world_model_error": 5.0, "uncertainty_mean": 0.3, "prev_uncertainty_mean": 0.1,
            "action": 3, "prev_action": 3, "visit_counter": vc, "player_pos": player.position}
    r1 = ir.compute(**base)
    vc.update(player.position)
    r2 = ir.compute(**base)
    # novelty decays with visits, repetition penalty present
    assert r2["novelty"] < r1["novelty"]
    assert r1["repetition"] > 0
    # no repetition when actions differ
    r3 = ir.compute(**{**base, "prev_action": 4})
    assert r3["repetition"] == 0
    # information gain scales with uncertainty delta
    assert r1["information_gain"] > 0


def test_utility_map_excludes_walls(cfg_p1, schema_p1, env_p1):
    ir = IntrinsicReward(cfg_p1)
    tr = BeliefTracker(schema_p1)
    obs = env_p1.reset(seed=5)
    state = tr.track(None, obs, 0)
    util = ir.explore_utility_map(state, VisitCounter())
    assert util.shape == (10, 10)
    assert np.isneginf(util).any()  # walls excluded


def test_intrinsic_changes_goal_priorities(cfg_p1, schema_p1, env_p1):
    from open_player.motivation.goals import GoalManager
    from open_player.motivation.motivation import IntrinsicMotivation
    tr = BeliefTracker(schema_p1)
    obs = env_p1.reset(seed=5)
    state = tr.track(None, obs, 0)
    mot = IntrinsicMotivation(cfg_p1).compute(state)
    gm = GoalManager(cfg_p1)
    base_info = {"threat_level": 0.0, "intrinsic_novelty": 0.0, "uncertainty_mean": 0.0}
    boosted_info = {"threat_level": 0.0, "intrinsic_novelty": 1.0, "uncertainty_mean": 1.0}
    base = gm.generate_candidates(state, mot, base_info)
    boosted = gm.generate_candidates(state, mot, boosted_info)
    # uncertainty raises the information goal priority
    def info_priority(cands):
        for g in cands:
            if g.goal_type == "information":
                return g.priority
        return None
    assert info_priority(boosted) >= info_priority(base)
    # intrinsic novelty raises the exploration priority
    def expl_priority(cands):
        for g in cands:
            if g.goal_type == "exploration":
                return g.priority
        return None
    assert expl_priority(boosted) >= expl_priority(base)
