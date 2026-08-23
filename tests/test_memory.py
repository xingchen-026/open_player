"""Memory layer tests (working / episodic / procedural / semantic / spatial)."""
from __future__ import annotations

from open_player.core.types import Episode, EpisodeOutcome, Event, Goal, GoalType
from open_player.memory.episodic import EpisodicMemory
from open_player.memory.procedural import ProceduralMemory
from open_player.memory.semantic import SemanticMemory
from open_player.memory.spatial import SpatialMemoryStore
from open_player.memory.working import WorkingMemory


def test_working_memory():
    wm = WorkingMemory(capacity=4)
    for i in range(6):
        wm.add_event(Event(f"e{i}", "move", i))
    assert len(wm.recent_events) == 4
    goal = Goal("g1", GoalType.TASK.value, target="resource")
    wm.set_goal(goal)
    assert wm.snapshot()["goal"] == "g1"


def test_episodic_memory():
    em = EpisodicMemory(capacity=4)
    for i in range(6):
        em.store(Episode(f"ep{i}", goal=Goal(f"g{i}", GoalType.TASK.value), outcome=EpisodeOutcome.SUCCESS, start_t=0, end_t=10))
    assert len(em) == 4
    hits = em.query(goal_type=GoalType.TASK.value, outcome="success", limit=2)
    assert len(hits) == 2
    assert em.stats()["num_episodes"] == 4


def test_procedural_memory():
    pm = ProceduralMemory()
    pm.record("collect", True, reward=1.0)
    pm.record("collect", False, reward=0.0)
    assert pm.success_rate("collect") == 0.5
    assert pm.success_rate("unknown", default=0.7) == 0.7
    assert pm.stats("collect")["attempts"] == 2.0


def test_semantic_memory():
    sm = SemanticMemory()
    sm.observe("player", "collected", "resource-0")
    sm.observe("player", "collected", "resource-0")
    hits = sm.query(subject="player", predicate="collected")
    assert hits[0][1] == 2.0
    assert sm.stats()["num_facts"] == 1


def test_spatial_store(schema, state_pair):
    store = SpatialMemoryStore(schema.spatial.shape)
    store.update(state_pair[0])
    arr = store.get()
    assert arr.shape == schema.spatial.shape
    assert 0.0 <= store.novelty_mean() <= 1.0
