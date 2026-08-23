"""WorldState construction and manipulation.

The neural world is a batched tensor object (WorldState); the environment
produces structured observations; this module (with the schemas) owns the
conversion so no other module hard-codes tensor layouts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from open_player.core.schema import SchemaSet
from open_player.core.types import (
    BeliefState,
    EntityState,
    Observation,
    Relation,
    SpatialMemory,
    WorldState,
)

_EPS = 1e-6


def _empty_entity(i: int) -> EntityState:
    return EntityState(entity_id=f"empty-{i}", semantic_type="empty", status=0.0)


def _default_belief(entity: EntityState) -> BeliefState:
    if entity.semantic_type == "empty":
        return BeliefState(entity_id=entity.entity_id, existence_probability=0.0, identity_confidence=0.0, visibility_confidence=0.0, position_variance=1.0, velocity_variance=1.0, prediction_confidence=0.0)
    return BeliefState(entity_id=entity.entity_id, existence_probability=1.0, identity_confidence=0.9, visibility_confidence=0.8, position_variance=0.1, velocity_variance=0.1, prediction_confidence=0.5)


def make_temporal(t: int, last_reward: float = 0.0, change_flags: Sequence[float] = (), temporal_dim: int = 16) -> np.ndarray:
    """Deterministic temporal-state encoding.

    layout: [phase one-hot (4)] [change flags (4)] [last reward (1)]
    [time fraction (1)] [reserved zeros].
    """
    vec = np.zeros(temporal_dim, dtype=np.float32)
    vec[t % 4] = 1.0
    flags = list(change_flags)[:4]
    flags += [0.0] * (4 - len(flags))
    vec[4:8] = flags
    vec[8] = float(np.clip(last_reward, -1.0, 1.0))
    vec[9] = min(t / 1000.0, 1.0)
    return vec


def compute_relations(schema: SchemaSet, entities: List[EntityState], beliefs: List[BeliefState]) -> np.ndarray:
    """Compute the pairwise relation tensor [N, N, R] (vectorised)."""
    n = len(entities)
    R = schema.relation.R
    out = np.zeros((n, n, R), dtype=np.float32)
    if n == 0:
        return out

    pos = np.stack([np.asarray(e.position, dtype=np.float32) for e in entities])  # [N, 2]
    vel = np.stack([np.asarray(e.velocity, dtype=np.float32) for e in entities])  # [N, 2]
    sizes = np.asarray([e.size for e in entities], dtype=np.float32)  # [N]

    delta = pos[:, None, :] - pos[None, :, :]  # [N, N, 2]  (dst - src)
    dist = np.linalg.norm(delta, axis=-1)  # [N, N]
    direction = delta / np.maximum(dist[..., None], _EPS)
    relvel = (vel[None, :, :] - vel[:, None, :]) / max(schema.entity.max_speed, _EPS)
    overlap = np.maximum(0.0, 1.0 - dist / np.maximum((sizes[:, None] + sizes[None, :]) * 0.5, _EPS))
    vis = np.asarray(
        [[1.0 if beliefs[i].visibility_confidence > 0.5 and beliefs[j].visibility_confidence > 0.5 else 0.0 for j in range(n)] for i in range(n)],
        dtype=np.float32,
    )

    def put(name: str, values: np.ndarray) -> None:
        s, e = schema.relation._slices[name]
        out[:, :, s:e] = values.reshape(n, n, e - s)

    put("distance", (dist / max(schema.entity.world_size, _EPS))[:, :, None])
    put("direction", direction)
    put("relative_velocity", np.clip(relvel, -1.0, 1.0))
    put("overlap", np.clip(overlap, 0.0, 1.0)[:, :, None])
    put("visibility", vis[:, :, None])
    codes = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                codes[i, j] = schema.relation.semantic_code(entities[i].semantic_type, entities[j].semantic_type)
    put("semantic_relation", codes[:, :, None])
    return out


def make_uncertainty(
    belief_vecs: np.ndarray, spatial: np.ndarray, uncertainty_dim: int
) -> np.ndarray:
    """Aggregate belief means + spatial statistics into [D_uncertainty]."""
    vec = np.zeros(uncertainty_dim, dtype=np.float32)
    b = np.asarray(belief_vecs, dtype=np.float32)
    if b.size:
        means = b.mean(axis=0)
        vec[0] = means[0]
        vec[1] = means[2]
        vec[2] = means[4]
        vec[3] = means[5]
        vec[4] = means[6]
        vec[5] = means[7]
    sp = np.asarray(spatial, dtype=np.float32)
    if sp.size:
        vec[6] = float(sp.mean())
        vec[7] = float(sp.std())
    return vec


def build_world_state(
    schema: SchemaSet,
    entities: List[EntityState],
    beliefs: Optional[List[BeliefState]] = None,
    spatial: Optional[Any] = None,
    global_features: Optional[np.ndarray] = None,
    dynamics: Optional[np.ndarray] = None,
    temporal: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    t: int = 0,
    device: Any = "cpu",
    metadata: Optional[Dict[str, Any]] = None,
) -> WorldState:
    """Build a batched WorldState from structured inputs.

    * entities are sorted by id for stable slots and padded with "empty"
      entities up to max_entities;
    * beliefs default to "freshly seen" for real entities;
    * relations are computed pairwise from positions;
    * spatial is accepted as a SpatialMemory or a raw [C, H, W] array.
    """
    N = schema.max_entities
    ents = sorted(entities, key=lambda e: e.entity_id)[:N]
    while len(ents) < N:
        ents.append(_empty_entity(len(ents)))

    belief_map: Dict[str, BeliefState] = {}
    if beliefs:
        for b in beliefs:
            belief_map[b.entity_id] = b
    bels = [belief_map.get(e.entity_id, _default_belief(e)) for e in ents]

    entity_vecs = np.stack([schema.entity.encode(e) for e in ents]).astype(np.float32)
    belief_vecs = np.stack([schema.belief.encode(b) for b in bels]).astype(np.float32)
    relation_t = compute_relations(schema, ents, bels)

    if spatial is None:
        spatial_arr = np.zeros(schema.spatial.shape, dtype=np.float32)
    elif isinstance(spatial, SpatialMemory):
        spatial_arr = np.asarray(spatial.data, dtype=np.float32)
    else:
        spatial_arr = np.asarray(spatial, dtype=np.float32)
    if spatial_arr.shape != schema.spatial.shape:
        raise ValueError(f"spatial shape {spatial_arr.shape} != schema spatial shape {schema.spatial.shape}")

    D_g = schema.global_dim
    glob = np.zeros(D_g, dtype=np.float32)
    if global_features is not None:
        gf = np.asarray(global_features, dtype=np.float32).reshape(-1)
        glob[: min(D_g, gf.size)] = gf[:D_g]

    dyn = np.zeros(schema.dynamics_dim, dtype=np.float32) if dynamics is None else np.asarray(dynamics, dtype=np.float32).reshape(-1)[: schema.dynamics_dim]
    tmp = make_temporal(t, temporal_dim=schema.temporal_dim) if temporal is None else np.asarray(temporal, dtype=np.float32).reshape(-1)[: schema.temporal_dim]
    unc = make_uncertainty(belief_vecs, spatial_arr, schema.uncertainty_dim) if uncertainty is None else np.asarray(uncertainty, dtype=np.float32).reshape(-1)[: schema.uncertainty_dim]

    def T(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).unsqueeze(0).to(device=device)

    return WorldState(
        entity_ids=[e.entity_id for e in ents],
        semantic_types=[e.semantic_type for e in ents],
        entities_t=T(entity_vecs),
        beliefs_t=T(belief_vecs),
        relations_t=T(relation_t),
        spatial_t=T(spatial_arr),
        dynamics_t=T(dyn),
        temporal_t=T(tmp),
        global_t=T(glob),
        uncertainty_t=T(unc),
        t=t,
        schema=schema,
        metadata=metadata or {},
    )


def empty_world_state(schema: SchemaSet, batch: int = 1, device: Any = "cpu", t: int = 0) -> WorldState:
    """A fully-empty WorldState of the right shapes (padding / init helper)."""
    N = schema.max_entities
    ids = [f"empty-{i}" for i in range(N)]
    types = ["empty"] * N
    return WorldState(
        entity_ids=ids,
        semantic_types=types,
        entities_t=torch.zeros(batch, N, schema.entity.D_entity, device=device),
        beliefs_t=torch.zeros(batch, N, schema.belief.dim, device=device),
        relations_t=torch.zeros(batch, N, N, schema.relation.R, device=device),
        spatial_t=torch.zeros(batch, *schema.spatial.shape, device=device),
        dynamics_t=torch.zeros(batch, schema.dynamics_dim, device=device),
        temporal_t=torch.zeros(batch, schema.temporal_dim, device=device),
        global_t=torch.zeros(batch, schema.global_dim, device=device),
        uncertainty_t=torch.zeros(batch, schema.uncertainty_dim, device=device),
        t=t,
        schema=schema,
        metadata={},
    )


def stack_world_states(schema: SchemaSet, states: Sequence[WorldState], device: Any = "cpu") -> WorldState:
    """Stack a list of single-batch WorldStates into one batched WorldState."""
    if not states:
        raise ValueError("cannot stack an empty list of WorldStates")
    first = states[0]
    return WorldState(
        entity_ids=list(first.entity_ids),
        semantic_types=list(first.semantic_types),
        entities_t=torch.stack([s.entities_t[0] for s in states]).to(device=device, dtype=torch.float32),
        beliefs_t=torch.stack([s.beliefs_t[0] for s in states]).to(device=device, dtype=torch.float32),
        relations_t=torch.stack([s.relations_t[0] for s in states]).to(device=device, dtype=torch.float32),
        spatial_t=torch.stack([s.spatial_t[0] for s in states]).to(device=device, dtype=torch.float32),
        dynamics_t=torch.stack([s.dynamics_t[0] for s in states]).to(device=device, dtype=torch.float32),
        temporal_t=torch.stack([s.temporal_t[0] for s in states]).to(device=device, dtype=torch.float32),
        global_t=torch.stack([s.global_t[0] for s in states]).to(device=device, dtype=torch.float32),
        uncertainty_t=torch.stack([s.uncertainty_t[0] for s in states]).to(device=device, dtype=torch.float32),
        t=int(first.t),
        schema=schema,
        metadata=dict(first.metadata),
    )


def grid_channel(state: WorldState, channel: int, batch: int = 0) -> np.ndarray:
    """Downsample one spatial memory channel back to grid resolution.

    Uses max-pooling per grid cell so a wall anywhere inside a cell counts as
    a wall (safe for navigation checks).  Grid size comes from the state's
    metadata (set by the encoder/tracker from the raw observation).  Results
    are cached per WorldState (the arrays are treated as immutable).
    """
    cache = state.metadata.setdefault("_grid_cache", {})
    key = (batch, channel)
    if key in cache:
        return cache[key]
    sp = state.spatial_t[batch, channel].detach().cpu().numpy()
    gs = int(state.metadata.get("grid_size", state.schema.world_size if state.schema else sp.shape[0]))
    H, W = sp.shape
    # inverse of SpatialSchema._resize_2d: memory row m samples grid row
    # floor(m * (gs - 1) / (H - 1)); find each grid cell's memory footprint
    ys = np.clip(np.floor(np.linspace(0, gs, H)).astype(np.int64), 0, gs - 1)
    xs = np.clip(np.floor(np.linspace(0, gs, W)).astype(np.int64), 0, gs - 1)
    out = np.zeros((gs, gs), dtype=np.float32)
    for gy in range(gs):
        rows = np.where(ys == gy)[0]
        if rows.size == 0:
            continue
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        for gx in range(gs):
            cols = np.where(xs == gx)[0]
            if cols.size == 0:
                continue
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
            out[gy, gx] = sp[y0:y1, x0:x1].max()
    cache[key] = out
    return out


def entity_index(state: WorldState, entity_id: str) -> int:
    """Slot index of an entity id in the padded entity tensor."""
    try:
        return state.entity_ids.index(entity_id)
    except ValueError:
        return -1


def entities_by_type(state: WorldState, semantic_type: str, batch: int = 0) -> List[EntityState]:
    """Decoded entities of one semantic type (batch 0)."""
    return [e for e in state.entity_states(batch) if e.semantic_type == semantic_type]
