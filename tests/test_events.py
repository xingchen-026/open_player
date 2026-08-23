"""Event detection / graph / episode tests."""
from __future__ import annotations

from open_player.core.types import EpisodeOutcome, Event, EventType
from open_player.events.detector import HeuristicEventDetector
from open_player.events.graph import EventGraph
from open_player.memory.episodic import EpisodeBuilder


def test_appear_on_first_detection(state_pair):
    det = HeuristicEventDetector()
    events = det.detect(None, state_pair[0], {}, 0)
    types = [e.type for e in events]
    assert EventType.APPEAR.value in types


def test_move_and_collect_detection(state_pair):
    det = HeuristicEventDetector()
    s0, s1 = state_pair
    events = det.detect(s0, s1, {"collected_this_step": True, "collected_entity": "resource-0", "hp": 4, "threat_level": 0.0, "prev_threat": 0.0}, 1)
    types = [e.type for e in events]
    assert EventType.COLLECT.value in types


def test_damage_detection(state_pair):
    det = HeuristicEventDetector()
    events = det.detect(state_pair[0], state_pair[1], {"hp_delta": -1, "hp": 2, "threat_level": 0.5, "prev_threat": 0.0, "damage_from": "enemy-0"}, 1)
    types = [e.type for e in events]
    assert EventType.DAMAGE.value in types
    assert EventType.COLLISION.value in types


def test_event_graph_relations():
    g = EventGraph()
    e1 = g.add_event(Event("e1", "move", 1, entities=["player"]))
    e2 = g.add_event(Event("e2", "collect", 2, entities=["player"]))
    e3 = g.add_event(Event("e3", "damage", 3, entities=["player"]))
    assert e2.parent_event == "e1"
    assert len(g.parents("e2")) == 1
    g.connect_causal("e2", "e3")
    assert [p.event_id for p in g.parents("e3")] == ["e2", "e3"][:1] or [p.event_id for p in g.parents("e3")][0] == "e2"
    comp = g.compose(["e1", "e2"], composite_type="sequence")
    assert comp.type == "sequence"
    stats = g.stats()
    assert stats["num_events"] == 4


def test_episode_builder(state_pair):
    b = EpisodeBuilder("ep-1", None, state_pair[0], 0)
    b.on_event(Event("x", "move", 1))
    b.on_skill("collect")
    b.on_action(3)
    ep = b.finish(EpisodeOutcome.SUCCESS, state_pair[1], 5, {})
    assert ep.outcome == EpisodeOutcome.SUCCESS
    assert ep.length == 5
    assert ep.skill_trace == ["collect"]
    assert ep.action_trace == [3]
    assert len(ep.events) == 1
