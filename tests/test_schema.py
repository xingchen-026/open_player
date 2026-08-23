"""Schema tests: derived dims, encode/decode roundtrips."""
from __future__ import annotations

import numpy as np

from open_player.core.schema import SchemaSet, TypeRegistry
from open_player.core.types import BeliefState, EntityState, Relation


def test_entity_dim_is_derived_not_hardcoded(schema):
    assert schema.entity.D_entity == 42
    assert schema.belief.dim == 8
    assert schema.relation.R == 8
    assert schema.spatial.shape == (16, 32, 32)
    assert schema.max_entities == 32


def test_entity_encode_decode_roundtrip(schema):
    e = EntityState(
        entity_id="player",
        semantic_type="player",
        position=np.array([5.0, 6.0], dtype=np.float32),
        velocity=np.array([0.0, 1.0], dtype=np.float32),
        size=1.0,
        status=0.75,
    )
    vec = schema.entity.encode(e)
    assert vec.shape == (42,)
    dec = schema.entity.decode(vec, entity_id="player")
    assert dec.semantic_type == "player"
    np.testing.assert_allclose(dec.position, e.position, atol=1e-4)
    np.testing.assert_allclose(dec.velocity, e.velocity, atol=1e-4)
    assert abs(dec.status - 0.75) < 1e-4


def test_semantic_features_onehot(schema):
    vec = schema.entity.encode(EntityState("x", "enemy"))
    s, e = schema.entity.field_slice("semantic_features")
    onehot = vec[s:e]
    assert onehot.sum() == 1.0
    assert schema.entity.registry.decode_onehot(onehot) == "enemy"


def test_belief_encode_decode(schema):
    b = BeliefState("e", position_variance=0.3, velocity_variance=0.4, existence_probability=0.9, identity_confidence=0.8, visibility_confidence=0.7, prediction_confidence=0.6)
    vec = schema.belief.encode(b)
    assert vec.shape == (8,)
    dec = schema.belief.decode(vec, "e")
    assert abs(dec.position_variance - 0.3) < 1e-5
    assert abs(dec.velocity_variance - 0.4) < 1e-5
    assert abs(dec.existence_probability - 0.9) < 1e-5
    assert abs(dec.identity_confidence - 0.8) < 1e-5


def test_relation_encode_semantic_codes(schema):
    r = Relation(src_id="player", dst_id="enemy-0", distance=2.0, direction=np.array([1.0, 0.0]), relative_velocity=np.array([0.5, 0.0]), overlap=0.1, visibility=1.0, semantic_relation="threat")
    vec = schema.relation.encode(r)
    assert vec.shape == (8,)
    dec = schema.relation.decode(vec, "player", "enemy-0")
    assert dec.semantic_relation == "threat"
    assert schema.relation.semantic_code("player", "enemy") == 2.0


def test_type_registry_dynamic(schema):
    reg = TypeRegistry()
    i = reg.register("chest")
    assert reg.name(i) == "chest"
    assert reg.index("chest") == i
    onehot = reg.onehot("chest", 8)
    assert onehot[i] == 1.0 and onehot.sum() == 1.0
