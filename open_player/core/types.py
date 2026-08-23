"""Core data structures of Open Player (Phase 0).

This module defines the frozen public vocabulary of the project.  Nothing in
here knows about a concrete game: entities are described by a generic
"semantic_type" string, and all dense content lives in tensors whose layout
is managed by :mod:"open_player.core.schema" (never hard-coded here).

Two representations coexist on purpose:

* structured objects (EntityState, BeliefState, Relation, ...) - readable by
  rules, heuristics, events and planners;
* batched tensors ([B, N, D_entity], [B, N, N, R], [B, C, H, W], ...) - the
  canonical input of the neural modules.

WorldState owns both: it keeps the tensors plus the id/type lists needed to
decode them back to structured objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from open_player.core.schema import SchemaSet

# --------------------------------------------------------------------------- #
# Semantic vocabulary
# --------------------------------------------------------------------------- #
# A plain tuple on purpose: the TypeRegistry in core/schema assigns tensor
# indices at runtime, so new environments can register their own semantic
# types without touching this module.
DEFAULT_SEMANTIC_TYPES: Tuple[str, ...] = ("empty", "player", "enemy", "resource", "wall")


# --------------------------------------------------------------------------- #
# Entity
# --------------------------------------------------------------------------- #
@dataclass
class EntityState:
    """Structured description of one entity (no batch, no tensors).

    The batched tensor [B, N, D_entity] is derived from a list of these via
    :class:"open_player.core.schema.EntitySchema".
    """

    entity_id: str
    semantic_type: str = "empty"
    position: Optional[np.ndarray] = None  # [2] grid/world coordinates
    velocity: Optional[np.ndarray] = None  # [2]
    size: float = 1.0
    appearance: Optional[np.ndarray] = None  # [D_appearance], schema-driven
    semantic_features: Optional[np.ndarray] = None  # [D_semantic_features]
    dynamics_features: Optional[np.ndarray] = None  # [D_dynamics_features]
    status: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = np.zeros(2, dtype=np.float32)
        if self.velocity is None:
            self.velocity = np.zeros(2, dtype=np.float32)

    def copy(self) -> "EntityState":
        return EntityState(
            entity_id=self.entity_id,
            semantic_type=self.semantic_type,
            position=None if self.position is None else self.position.copy(),
            velocity=None if self.velocity is None else self.velocity.copy(),
            size=self.size,
            appearance=None if self.appearance is None else self.appearance.copy(),
            semantic_features=None if self.semantic_features is None else self.semantic_features.copy(),
            dynamics_features=None if self.dynamics_features is None else self.dynamics_features.copy(),
            status=self.status,
            metadata=dict(self.metadata),
        )


# --------------------------------------------------------------------------- #
# Belief
# --------------------------------------------------------------------------- #
@dataclass
class BeliefState:
    """Uncertainty / confidence attached to one entity (separate from state).

    Tensor layout [B, N, D_belief] with D_belief = 8:

    =========================  ============================================
    dims                       meaning
    =========================  ============================================
    [0:2]                      position_variance (x, y)
    [2:4]                      velocity_variance (x, y)
    [4]                        existence_probability
    [5]                        identity_confidence
    [6]                        visibility_confidence
    [7]                        prediction_confidence
    =========================  ============================================
    """

    entity_id: str
    position_variance: float = 1.0
    velocity_variance: float = 1.0
    existence_probability: float = 0.0
    identity_confidence: float = 0.0
    visibility_confidence: float = 0.0
    prediction_confidence: float = 0.5


# --------------------------------------------------------------------------- #
# Relation
# --------------------------------------------------------------------------- #
@dataclass
class Relation:
    """Pairwise relation between two entities.

    Tensor layout: [B, N, N, R]; R is derived from the relation schema
    (distance, direction, relative_velocity, overlap, visibility,
    semantic_relation).
    """

    src_id: str
    dst_id: str
    distance: float = 0.0
    direction: Optional[np.ndarray] = None  # [2], dst - src (normalised-ish)
    relative_velocity: Optional[np.ndarray] = None  # [2]
    overlap: float = 0.0
    visibility: float = 0.0
    semantic_relation: str = "none"  # none | interact | threat | kin | ...

    def __post_init__(self) -> None:
        if self.direction is None:
            self.direction = np.zeros(2, dtype=np.float32)
        if self.relative_velocity is None:
            self.relative_velocity = np.zeros(2, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Spatial memory
# --------------------------------------------------------------------------- #
@dataclass
class SpatialMemory:
    """Grid memory, [C, H, W].  NOT entity memory.

    Phase 0 defaults: C=16, H=32, W=32.  Channels may hold occupancy, threat,
    novelty, resource likelihood, navigation memory, ...
    """

    data: np.ndarray  # [C, H, W]
    channels: List[str] = field(default_factory=list)
    resolution: float = 1.0

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """Structured observation produced by an environment (Phase 0: synthetic).

    A real vision encoder will later produce the same fields from pixels;
    Phase 0 uses the DummyVisionEncoder to turn this into a WorldState.
    """

    entities: List[EntityState]
    spatial: np.ndarray  # [C_obs, H, W] raw channels
    global_features: np.ndarray  # [D_obs_global]
    t: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# WorldState - the core structured state
# --------------------------------------------------------------------------- #
@dataclass
class WorldState:
    """The central structured state of the system.

    Never compressed into a single vector: entities, beliefs, relations,
    spatial memory, dynamics, temporal, global and uncertainty tensors all
    stay explicit.  All tensors carry a batch dimension.
    """

    entity_ids: List[str]
    semantic_types: List[str]
    entities_t: Tensor  # [B, N, D_entity]
    beliefs_t: Tensor  # [B, N, D_belief]
    relations_t: Tensor  # [B, N, N, R]
    spatial_t: Tensor  # [B, C, H, W]
    dynamics_t: Tensor  # [B, D_dyn]
    temporal_t: Tensor  # [B, D_temporal]
    global_t: Tensor  # [B, D_global]
    uncertainty_t: Tensor  # [B, D_uncertainty]
    t: int = 0
    schema: Optional["SchemaSet"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- shape helpers --------------------------------------------------- #
    @property
    def batch_size(self) -> int:
        return int(self.entities_t.shape[0])

    @property
    def num_entities(self) -> int:
        return int(self.entities_t.shape[1])

    @property
    def device(self) -> Any:
        return self.entities_t.device

    @property
    def dtype(self) -> Any:
        return self.entities_t.dtype

    # -- tensor helpers -------------------------------------------------- #
    def to(self, device: Any, dtype: Any = None) -> "WorldState":
        """Move every tensor to device (in place, returns self for chaining)."""
        self.entities_t = self.entities_t.to(device=device, dtype=dtype)
        self.beliefs_t = self.beliefs_t.to(device=device, dtype=dtype)
        self.relations_t = self.relations_t.to(device=device, dtype=dtype)
        self.spatial_t = self.spatial_t.to(device=device, dtype=dtype)
        self.dynamics_t = self.dynamics_t.to(device=device, dtype=dtype)
        self.temporal_t = self.temporal_t.to(device=device, dtype=dtype)
        self.global_t = self.global_t.to(device=device, dtype=dtype)
        self.uncertainty_t = self.uncertainty_t.to(device=device, dtype=dtype)
        return self

    def detach(self) -> "WorldState":
        """Return a detached copy sharing ids/types/metadata."""
        return WorldState(
            entity_ids=list(self.entity_ids),
            semantic_types=list(self.semantic_types),
            entities_t=self.entities_t.detach(),
            beliefs_t=self.beliefs_t.detach(),
            relations_t=self.relations_t.detach(),
            spatial_t=self.spatial_t.detach(),
            dynamics_t=self.dynamics_t.detach(),
            temporal_t=self.temporal_t.detach(),
            global_t=self.global_t.detach(),
            uncertainty_t=self.uncertainty_t.detach(),
            t=self.t,
            schema=self.schema,
            metadata=dict(self.metadata),
        )

    def compact(self) -> "WorldState":
        """Cast bulky tensors (relations, spatial) to float16 for storage.

        Used by the replay buffer to keep CPU memory bounded.  Training casts
        back to float32 when sampling.
        """
        out = self.detach().to("cpu")
        out.relations_t = out.relations_t.half()
        out.spatial_t = out.spatial_t.half()
        return out

    # -- structured decoders -------------------------------------------- #
    def entity_states(self, batch: int = 0) -> List[EntityState]:
        """Decode batch item into structured EntityState objects (cached)."""
        if self.schema is None:
            raise RuntimeError("WorldState has no schema attached; cannot decode entities")
        cache = self.metadata.setdefault("_entity_cache", {})
        if batch in cache:
            return cache[batch]
        vecs = self.entities_t[batch].detach().cpu().numpy()
        out: List[EntityState] = []
        for i, vec in enumerate(vecs):
            e = self.schema.entity.decode(
                vec,
                entity_id=self.entity_ids[i] if i < len(self.entity_ids) else None,
                semantic_type=self.semantic_types[i] if i < len(self.semantic_types) else None,
            )
            out.append(e)
        cache[batch] = out
        return out

    def belief_states(self, batch: int = 0) -> List[BeliefState]:
        """Decode batch item into structured BeliefState objects."""
        if self.schema is None:
            raise RuntimeError("WorldState has no schema attached; cannot decode beliefs")
        vecs = self.beliefs_t[batch].detach().cpu().numpy()
        out: List[BeliefState] = []
        for i, vec in enumerate(vecs):
            eid = self.entity_ids[i] if i < len(self.entity_ids) else f"entity-{i}"
            out.append(self.schema.belief.decode(vec, entity_id=eid))
        return out

    def summary(self) -> Dict[str, Any]:
        """Compact, serialisable snapshot (used by logs and checkpoints)."""
        return {
            "t": self.t,
            "batch_size": self.batch_size,
            "num_entities": self.num_entities,
            "entity_ids": list(self.entity_ids),
            "semantic_types": list(self.semantic_types),
            "entity_shape": [int(s) for s in self.entities_t.shape],
            "spatial_shape": [int(s) for s in self.spatial_t.shape],
        }


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    """Primitive semantic event types supported by Phase 0."""

    APPEAR = "appear"
    DISAPPEAR = "disappear"
    MOVE = "move"
    APPROACH = "approach"
    COLLISION = "collision"
    DAMAGE = "damage"
    DEATH = "death"
    COLLECT = "collect"
    ENTER = "enter"
    EXIT = "exit"
    THREAT_INCREASE = "threat_increase"
    THREAT_DECREASE = "threat_decrease"
    STATE_CHANGE = "state_change"


#: Events that indicate a *meaningful* world change (target of the
#: change-prediction head; also used for episode boundary heuristics).
SEMANTIC_EVENT_TYPES: Tuple[str, ...] = (
    EventType.APPEAR.value,
    EventType.DISAPPEAR.value,
    EventType.COLLISION.value,
    EventType.DAMAGE.value,
    EventType.DEATH.value,
    EventType.COLLECT.value,
    EventType.THREAT_INCREASE.value,
    EventType.THREAT_DECREASE.value,
)


@dataclass
class Event:
    """A primitive event; composite events and episodes live above it."""

    event_id: str
    type: str  # one of EventType values
    timestamp: int
    entities: List[str] = field(default_factory=list)
    location: Optional[np.ndarray] = None  # [2]
    confidence: float = 1.0
    parent_event: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Goal
# --------------------------------------------------------------------------- #
class GoalType(str, Enum):
    TASK = "task"
    EXPLORATION = "exploration"
    SURVIVAL = "survival"
    LEARNING = "learning"
    SKILL_IMPROVEMENT = "skill_improvement"
    INFORMATION = "information"


@dataclass
class Goal:
    """A selected (or candidate) goal.

    Goals come from three inputs: external goals, intrinsic motivation and
    the current situation.  The GoalManager scores candidates and selects one.
    """

    goal_id: str
    goal_type: str  # GoalType value
    target: str = ""  # semantic type ("resource") or region/location key
    location: Optional[np.ndarray] = None  # [2]
    priority: float = 0.5
    source: str = "intrinsic"  # external | intrinsic:<drive>
    status: str = "active"  # active | succeeded | failed | abandoned
    progress: float = 0.0
    created_t: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Action
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    """One action issued to an environment.

    "index" addresses the environment's action space (discrete in Phase 0);
    "params" is reserved for continuous / hybrid spaces.
    """

    name: str
    index: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def discrete(self) -> int:
        return int(self.index)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Action(name={self.name!r}, index={self.index})"


# --------------------------------------------------------------------------- #
# Episode
# --------------------------------------------------------------------------- #
class EpisodeOutcome(str, Enum):
    ONGOING = "ongoing"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass
class Episode:
    """An episode: event-boundary + goal hybrid segmentation unit."""

    episode_id: str
    start_state: Optional["WorldState"] = None
    goal: Optional["Goal"] = None
    events: List["Event"] = field(default_factory=list)
    skill_trace: List[str] = field(default_factory=list)
    action_trace: List[int] = field(default_factory=list)
    outcome: EpisodeOutcome = EpisodeOutcome.ONGOING
    failure_analysis: Dict[str, Any] = field(default_factory=dict)
    end_state: Optional["WorldState"] = None
    start_t: int = 0
    end_t: int = 0

    @property
    def length(self) -> int:
        return max(0, self.end_t - self.start_t) if self.end_t > 0 else 0
