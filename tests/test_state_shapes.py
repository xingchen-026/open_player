"""WorldState construction + tensor shape tests."""
from __future__ import annotations

import numpy as np
import torch

from open_player.core.state import build_world_state, empty_world_state, stack_world_states
from open_player.core.types import EntityState, SpatialMemory


def test_world_state_shapes(schema):
    ents = [
        EntityState("player", "player", np.array([2.0, 2.0])),
        EntityState("enemy-0", "enemy", np.array([6.0, 6.0])),
    ]
    sp = SpatialMemory(data=np.zeros(schema.spatial.shape), channels=list(schema.spatial.MEMORY_CHANNELS))
    ws = build_world_state(schema, ents, spatial=sp, global_features=np.zeros(4, dtype=np.float32), t=0)
    N = schema.max_entities
    assert ws.entities_t.shape == (1, N, 42)
    assert ws.beliefs_t.shape == (1, N, 8)
    assert ws.relations_t.shape == (1, N, N, 8)
    assert ws.spatial_t.shape == (1, 16, 32, 32)
    assert ws.dynamics_t.shape == (1, schema.dynamics_dim)
    assert ws.temporal_t.shape == (1, schema.temporal_dim)
    assert ws.global_t.shape == (1, schema.global_dim)
    assert ws.uncertainty_t.shape == (1, schema.uncertainty_dim)
    assert len(ws.entity_ids) == N
    assert "player" in ws.entity_ids


def test_entity_padding_and_decode(schema):
    ents = [EntityState("player", "player", np.array([1.0, 1.0]))]
    ws = build_world_state(schema, ents)
    assert ws.entity_ids[1].startswith("empty")
    states = ws.entity_states(0)
    player = [e for e in states if e.semantic_type == "player"]
    assert len(player) == 1
    np.testing.assert_allclose(player[0].position, [1.0, 1.0], atol=1e-4)


def test_relations_computed(schema):
    ents = [
        EntityState("player", "player", np.array([0.0, 0.0])),
        EntityState("enemy-0", "enemy", np.array([3.0, 4.0])),
    ]
    ws = build_world_state(schema, ents)
    ip = ws.entity_ids.index("player")
    ie = ws.entity_ids.index("enemy-0")
    dist = float(ws.relations_t[0, ip, ie, 0])
    assert abs(dist - 5.0 / schema.world_size) < 1e-3
    # semantic threat code between player and enemy
    s, e = schema.relation.field_slice if hasattr(schema.relation, "field_slice") else (None, None)
    rel_slice = schema.relation._slices["semantic_relation"]
    assert float(ws.relations_t[0, ip, ie, rel_slice[0]]) == 2.0


def test_empty_and_stack(schema):
    ws = empty_world_state(schema, batch=2)
    assert ws.entities_t.shape == (2, schema.max_entities, 42)
    single = empty_world_state(schema, batch=1)
    stacked = stack_world_states(schema, [single, single])
    assert stacked.entities_t.shape == (2, schema.max_entities, 42)


def test_compact_keeps_shapes(schema):
    ents = [EntityState("player", "player", np.array([1.0, 1.0]))]
    ws = build_world_state(schema, ents)
    c = ws.compact()
    assert c.relations_t.dtype == torch.float16
    assert c.spatial_t.dtype == torch.float16
    assert c.entities_t.dtype == torch.float32
