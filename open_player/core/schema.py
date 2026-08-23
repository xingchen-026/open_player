"""Schema-driven encoding between structured objects and tensors.

Phase 0 rule: tensor dims are NEVER hard-coded in the modules that consume
them.  EntitySchema / BeliefSchema / RelationSchema / SpatialSchema define
the layouts; D_entity (42 by default), D_belief (8), R (8) and the spatial
shape (16 x 32 x 32) all fall out of the configuration.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from open_player.core.types import (
    BeliefState,
    EntityState,
    Relation,
    DEFAULT_SEMANTIC_TYPES,
)


@dataclass
class FieldSpec:
    """One named field of a tensor layout."""

    name: str
    dim: int


def _coerce_field(f: Any) -> FieldSpec:
    """Accept FieldSpec, (name, dim) tuples or {'name': ..., 'dim': ...} dicts."""
    if isinstance(f, FieldSpec):
        return f
    if isinstance(f, dict):
        return FieldSpec(str(f["name"]), int(f["dim"]))
    return FieldSpec(str(f[0]), int(f[1]))


# --------------------------------------------------------------------------- #
# Semantic type registry
# --------------------------------------------------------------------------- #
class TypeRegistry:
    """Maps semantic type strings to tensor indices (dynamic, extensible)."""

    def __init__(self, initial: Iterable[str] = DEFAULT_SEMANTIC_TYPES) -> None:
        self._index: Dict[str, int] = {}
        self._names: List[str] = []
        for name in initial:
            self.register(name)

    def register(self, name: str) -> int:
        if name not in self._index:
            self._index[name] = len(self._names)
            self._names.append(name)
        return self._index[name]

    def index(self, name: str) -> int:
        return self._index.get(name, self.register(name))

    def name(self, idx: int) -> str:
        return self._names[idx]

    @property
    def count(self) -> int:
        return len(self._names)

    def onehot(self, name: str, dim: int) -> np.ndarray:
        """One-hot of a semantic type in a vector of length dim."""
        vec = np.zeros(dim, dtype=np.float32)
        idx = self.index(name)
        if idx < dim:
            vec[idx] = 1.0
        return vec

    def decode_onehot(self, vec: np.ndarray) -> str:
        idx = int(np.argmax(vec))
        if idx < self.count and float(vec[idx]) > 0.5:
            return self._names[idx]
        return "empty"


# --------------------------------------------------------------------------- #
# Appearance
# --------------------------------------------------------------------------- #
def appearance_vector(entity_id: str, dim: int) -> np.ndarray:
    """Deterministic pseudo-random appearance features for an entity id.

    Phase 0 has no real vision: a stable per-id vector keeps the pipeline
    trainable and deterministic without any game-specific hard-coding.
    """
    digest = hashlib.sha256(entity_id.encode("utf-8")).digest()
    values = np.frombuffer(digest[: 4 * dim], dtype=np.uint32).astype(np.float32)
    if values.size < dim:  # pragma: no cover - defensive
        values = np.resize(values, dim)
    values = values[:dim]
    maxv = float(2 ** 32 - 1)
    return np.clip(values / maxv, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Entity schema
# --------------------------------------------------------------------------- #
class EntitySchema:
    """Encodes EntityState <-> [D_entity] vectors.

    Field-specific conventions:

    * position / world_size           -> [0, 1]^2
    * velocity / max_speed            -> [-1, 1]^2
    * size / max_size                 -> [0, 1]
    * appearance                      -> [0, 1]^D (deterministic per entity)
    * semantic_features               -> one-hot of the semantic type
    * dynamics_features               -> task-provided, passed through
    * status                          -> [0, 1]
    """

    def __init__(
        self,
        fields: Sequence[Any],
        world_size: float = 12.0,
        max_speed: float = 1.0,
        max_size: float = 3.0,
        max_entities: int = 32,
        registry: Optional[TypeRegistry] = None,
    ) -> None:
        self.fields: List[FieldSpec] = [_coerce_field(f) for f in fields]
        self.world_size = float(world_size)
        self.max_speed = float(max_speed)
        self.max_size = float(max_size)
        self.max_entities = int(max_entities)
        self.registry = registry if registry is not None else TypeRegistry()
        self._slices: Dict[str, Tuple[int, int]] = {}
        start = 0
        for f in self.fields:
            self._slices[f.name] = (start, start + f.dim)
            start += f.dim

    @property
    def D_entity(self) -> int:
        return sum(f.dim for f in self.fields)

    def field_slice(self, name: str) -> Tuple[int, int]:
        if name not in self._slices:
            raise KeyError(f"unknown entity field '{name}'; available: {sorted(self._slices)}")
        return self._slices[name]

    def field_dim(self, name: str) -> int:
        s, e = self.field_slice(name)
        return e - s

    # -- encode ---------------------------------------------------------- #
    def encode(self, entity: EntityState) -> np.ndarray:
        vec = np.zeros(self.D_entity, dtype=np.float32)
        pos = np.asarray(entity.position if entity.position is not None else [0.0, 0.0], dtype=np.float32)
        vel = np.asarray(entity.velocity if entity.velocity is not None else [0.0, 0.0], dtype=np.float32)

        self._set(vec, "position", pos / self.world_size)
        self._set(vec, "velocity", np.clip(vel / max(self.max_speed, 1e-6), -1.0, 1.0))
        self._set(vec, "size", [min(max(entity.size / max(self.max_size, 1e-6), 0.0), 1.0)])

        app = entity.appearance
        if app is None:
            app = appearance_vector(entity.entity_id, self.field_dim("appearance"))
        self._set(vec, "appearance", np.asarray(app, dtype=np.float32)[: self.field_dim("appearance")])

        sem = entity.semantic_features
        if sem is None:
            sem = self.registry.onehot(entity.semantic_type, self.field_dim("semantic_features"))
        self._set(vec, "semantic_features", np.asarray(sem, dtype=np.float32)[: self.field_dim("semantic_features")])

        dyn = entity.dynamics_features
        if dyn is None:
            dyn = np.zeros(self.field_dim("dynamics_features"), dtype=np.float32)
        self._set(vec, "dynamics_features", np.asarray(dyn, dtype=np.float32)[: self.field_dim("dynamics_features")])

        self._set(vec, "status", [float(np.clip(entity.status, 0.0, 1.0))])
        return vec

    def _set(self, vec: np.ndarray, name: str, value: Any) -> None:
        s, e = self.field_slice(name)
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        n = min(e - s, arr.size)
        vec[s : s + n] = arr[:n]

    # -- decode ---------------------------------------------------------- #
    def decode(self, vec: np.ndarray, entity_id: Optional[str] = None, semantic_type: Optional[str] = None) -> EntityState:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if entity_id is None:
            entity_id = "decoded"

        def _take(name: str) -> np.ndarray:
            s, e = self.field_slice(name)
            return vec[s:e]

        pos = _take("position") * self.world_size
        vel = _take("velocity") * self.max_speed
        size = float(_take("size")[0] * self.max_size)
        sem_vec = _take("semantic_features")
        if semantic_type is None:
            semantic_type = self.registry.decode_onehot(sem_vec)
        return EntityState(
            entity_id=entity_id,
            semantic_type=semantic_type,
            position=pos.astype(np.float32),
            velocity=vel.astype(np.float32),
            size=size,
            appearance=_take("appearance").copy(),
            semantic_features=sem_vec.copy(),
            dynamics_features=_take("dynamics_features").copy(),
            status=float(_take("status")[0]),
        )

    def decode_position(self, vec: np.ndarray) -> np.ndarray:
        s, e = self.field_slice("position")
        return np.asarray(vec, dtype=np.float32).reshape(-1)[s:e] * self.world_size


# --------------------------------------------------------------------------- #
# Belief schema
# --------------------------------------------------------------------------- #
class BeliefSchema:
    """Encodes BeliefState <-> [D_belief] vectors (D_belief = 8).

    dims [0:2] position_variance, [2:4] velocity_variance,
    [4] existence_probability, [5] identity_confidence,
    [6] visibility_confidence, [7] prediction_confidence.
    """

    D_BELIEF = 8
    SLICES: Dict[str, Tuple[int, int]] = {
        "position_variance": (0, 2),
        "velocity_variance": (2, 4),
        "existence_probability": (4, 5),
        "identity_confidence": (5, 6),
        "visibility_confidence": (6, 7),
        "prediction_confidence": (7, 8),
    }

    def __init__(self, dim: int = 8) -> None:
        self.dim = int(dim)
        if self.dim < BeliefSchema.D_BELIEF:
            raise ValueError("belief dim must be >= 8")

    def encode(self, belief: BeliefState) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = vec[1] = float(belief.position_variance)
        vec[2] = vec[3] = float(belief.velocity_variance)
        vec[4] = float(belief.existence_probability)
        vec[5] = float(belief.identity_confidence)
        vec[6] = float(belief.visibility_confidence)
        vec[7] = float(belief.prediction_confidence)
        return vec

    def decode(self, vec: np.ndarray, entity_id: str = "") -> BeliefState:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        return BeliefState(
            entity_id=entity_id,
            position_variance=float(v[0]),
            velocity_variance=float(v[2]),
            existence_probability=float(v[4]),
            identity_confidence=float(v[5]),
            visibility_confidence=float(v[6]),
            prediction_confidence=float(v[7]),
        )


# --------------------------------------------------------------------------- #
# Relation schema
# --------------------------------------------------------------------------- #
class RelationSchema:
    """Encodes Relation <-> [R] vectors (default R = 8)."""

    CODES: Dict[str, float] = {"none": 0.0, "interact": 1.0, "threat": 2.0, "kin": 3.0}
    CODE_NAMES: Dict[float, str] = {v: k for k, v in CODES.items()}
    DEFAULT_PAIR_RULES: Tuple[Tuple[Tuple[str, str], str], ...] = (
        (("player", "resource"), "interact"),
        (("resource", "player"), "interact"),
        (("player", "enemy"), "threat"),
        (("enemy", "player"), "threat"),
        (("enemy", "enemy"), "kin"),
        (("resource", "resource"), "kin"),
        (("wall", "player"), "kin"),
        (("player", "wall"), "kin"),
    )

    def __init__(self, fields: Sequence[Any], world_size: float = 12.0, max_speed: float = 1.0) -> None:
        self.fields: List[FieldSpec] = [_coerce_field(f) for f in fields]
        self.world_size = float(world_size)
        self.max_speed = float(max_speed)
        self.pair_rules: List[Tuple[Tuple[str, str], str]] = list(self.DEFAULT_PAIR_RULES)
        self._slices: Dict[str, Tuple[int, int]] = {}
        start = 0
        for f in self.fields:
            self._slices[f.name] = (start, start + f.dim)
            start += f.dim

    @property
    def R(self) -> int:
        return sum(f.dim for f in self.fields)

    def semantic_code(self, type_a: str, type_b: str) -> float:
        if type_a == type_b and type_a != "empty":
            return self.CODES["kin"]
        for (a, b), rel in self.pair_rules:
            if a == type_a and b == type_b:
                return self.CODES.get(rel, 0.0)
        return 0.0

    def semantic_name(self, code: float) -> str:
        return self.CODE_NAMES.get(float(round(code)), "none")

    def encode(self, rel: Relation) -> np.ndarray:
        vec = np.zeros(self.R, dtype=np.float32)
        s, e = self._slices["distance"]
        vec[s:e] = [min(rel.distance / max(self.world_size, 1e-6), 1.0)]
        s, e = self._slices["direction"]
        vec[s:e] = np.asarray(rel.direction, dtype=np.float32)
        s, e = self._slices["relative_velocity"]
        vec[s:e] = np.clip(np.asarray(rel.relative_velocity, dtype=np.float32) / max(self.max_speed, 1e-6), -1, 1)
        s, e = self._slices["overlap"]
        vec[s:e] = [min(max(rel.overlap, 0.0), 1.0)]
        s, e = self._slices["visibility"]
        vec[s:e] = [min(max(rel.visibility, 0.0), 1.0)]
        s, e = self._slices["semantic_relation"]
        vec[s:e] = [self.CODES.get(rel.semantic_relation, 0.0)]
        return vec

    def decode(self, vec: np.ndarray, src_id: str = "", dst_id: str = "") -> Relation:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        def take(name: str) -> np.ndarray:
            s, e = self._slices[name]
            return v[s:e]
        return Relation(
            src_id=src_id,
            dst_id=dst_id,
            distance=float(take("distance")[0]) * self.world_size,
            direction=take("direction").copy(),
            relative_velocity=take("relative_velocity").copy() * self.max_speed,
            overlap=float(take("overlap")[0]),
            visibility=float(take("visibility")[0]),
            semantic_relation=self.semantic_name(float(take("semantic_relation")[0])),
        )


# --------------------------------------------------------------------------- #
# Spatial schema
# --------------------------------------------------------------------------- #
class SpatialSchema:
    """Layout of the spatial memory tensor [C, H, W] and the raw observation
    channels, plus the Phase 0 mapping between them."""

    MEMORY_CHANNELS: Tuple[str, ...] = (
        "occupancy", "wall", "threat", "novelty", "resource", "navigation",
        "unknown", "visited", "energy", "goal_salience", "history", "proximity",
        "free0", "free1", "free2", "free3",
    )
    RAW_CHANNELS: Tuple[str, ...] = (
        "occupancy", "wall", "resource", "enemy", "player", "visited", "unknown", "threat",
    )

    def __init__(self, channels: int = 16, height: int = 32, width: int = 32) -> None:
        self.C = int(channels)
        self.H = int(height)
        self.W = int(width)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.C, self.H, self.W)

    def channel_index(self, name: str) -> int:
        return self.MEMORY_CHANNELS.index(name)

    def from_raw(self, raw: np.ndarray, raw_channels: Optional[Sequence[str]] = None) -> np.ndarray:
        """Map raw observation channels [C_obs, H, W] into memory channels [C, H, W]."""
        if raw_channels is None:
            raw_channels = list(self.RAW_CHANNELS)
        raw = np.asarray(raw, dtype=np.float32)
        out = np.zeros(self.shape, dtype=np.float32)
        lookup = {name: i for i, name in enumerate(raw_channels)}
        h, w = raw.shape[1], raw.shape[2]

        def grab(name: str) -> np.ndarray:
            if name in lookup:
                return self._resize_2d(raw[lookup[name]], self.H, self.W)
            return np.zeros((self.H, self.W), dtype=np.float32)

        out[self.channel_index("occupancy")] = grab("occupancy")
        out[self.channel_index("wall")] = grab("wall")
        out[self.channel_index("threat")] = grab("threat")
        visited = grab("visited")
        out[self.channel_index("visited")] = visited
        out[self.channel_index("novelty")] = 1.0 - visited
        out[self.channel_index("resource")] = grab("resource")
        out[self.channel_index("navigation")] = visited * 0.5
        out[self.channel_index("unknown")] = grab("unknown")
        return out

    @staticmethod
    def _resize_2d(a: np.ndarray, H: int, W: int) -> np.ndarray:
        h, w = a.shape
        ys = np.floor(np.linspace(0, h, H, endpoint=False)).astype(np.int64)
        xs = np.floor(np.linspace(0, w, W, endpoint=False)).astype(np.int64)
        return a[np.ix_(ys, xs)]


# --------------------------------------------------------------------------- #
# SchemaSet
# --------------------------------------------------------------------------- #
@dataclass
class SchemaSet:
    """All tensor layouts used by a WorldState, in one place."""

    entity: EntitySchema
    belief: BeliefSchema
    relation: RelationSchema
    spatial: SpatialSchema
    max_entities: int
    dynamics_dim: int
    temporal_dim: int
    global_dim: int
    uncertainty_dim: int
    world_size: int

    @classmethod
    def from_config(cls, cfg: Any) -> "SchemaSet":
        sc = cfg.schema if hasattr(cfg, "schema") else cfg
        world_size = int(sc["world_size"]) if isinstance(sc, dict) else int(sc.world_size)
        max_entities = int(sc["max_entities"]) if isinstance(sc, dict) else int(sc.max_entities)
        entity = EntitySchema(
            fields=sc["entity_fields"] if isinstance(sc, dict) else sc.entity_fields,
            world_size=world_size,
            max_speed=float(sc["max_speed"]) if isinstance(sc, dict) else float(sc.max_speed),
            max_size=float(sc["max_size"]) if isinstance(sc, dict) else float(sc.max_size),
            max_entities=max_entities,
        )
        belief = BeliefSchema(dim=int(sc["belief_dim"]) if isinstance(sc, dict) else int(sc.belief_dim))
        relation = RelationSchema(
            fields=sc["relation_fields"] if isinstance(sc, dict) else sc.relation_fields,
            world_size=world_size,
            max_speed=float(sc["max_speed"]) if isinstance(sc, dict) else float(sc.max_speed),
        )
        sp = sc["spatial"] if isinstance(sc, dict) else sc.spatial
        spatial = SpatialSchema(channels=int(sp["channels"]), height=int(sp["height"]), width=int(sp["width"]))
        return cls(
            entity=entity,
            belief=belief,
            relation=relation,
            spatial=spatial,
            max_entities=max_entities,
            dynamics_dim=int(sc["dynamics_dim"]) if isinstance(sc, dict) else int(sc.dynamics_dim),
            temporal_dim=int(sc["temporal_dim"]) if isinstance(sc, dict) else int(sc.temporal_dim),
            global_dim=int(sc["global_dim"]) if isinstance(sc, dict) else int(sc.global_dim),
            uncertainty_dim=int(sc["uncertainty_dim"]) if isinstance(sc, dict) else int(sc.uncertainty_dim),
            world_size=world_size,
        )
