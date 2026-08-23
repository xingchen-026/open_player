"""BeliefTracker: observation -> WorldState with belief update.

Performs the Belief Update step of the architecture loop: newly seen
entities are trusted, unobserved entities decay in existence and grow in
variance, and everything is re-encoded into the next WorldState.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from open_player.core.schema import SchemaSet
from open_player.core.state import build_world_state
from open_player.core.types import BeliefState, EntityState, Observation, SpatialMemory, WorldState
from open_player.tracking.association import associate_entities


class BeliefTracker:
    """Tracks entities across observations and maintains beliefs."""

    def __init__(
        self,
        schema: SchemaSet,
        device: Any = "cpu",
        existence_decay: float = 0.85,
        drop_threshold: float = 0.1,
        variance_growth: float = 0.2,
        max_assoc_dist: float = 2.0,
    ) -> None:
        self.schema = schema
        self.device = device
        self.existence_decay = float(existence_decay)
        self.drop_threshold = float(drop_threshold)
        self.variance_growth = float(variance_growth)
        self.max_assoc_dist = float(max_assoc_dist)

    def track(self, prev: Optional[WorldState], observation: Observation, t: int = 0) -> WorldState:
        """Build the next WorldState from an observation, fusing with prev."""
        if prev is None:
            return self._fresh(observation, t)
        prev_entities = {e.entity_id: e for e in prev.entity_states(0) if e.semantic_type != "empty"}
        prev_beliefs = {b.entity_id: b for b in prev.belief_states(0) if b.entity_id in prev_entities}

        curr_entities = list(observation.entities)
        matches = associate_entities(prev_entities, curr_entities, max_dist=self.max_assoc_dist)

        # Entities present now (fresh observation state + identity confidence)
        merged: Dict[str, EntityState] = {}
        beliefs: Dict[str, BeliefState] = {}
        for cent in curr_entities:
            eid = cent.entity_id
            merged[eid] = cent.copy()
            pb = prev_beliefs.get(eid)
            pred_conf = pb.prediction_confidence if pb is not None else 0.5
            matched_prev = any(pid == eid and m is not None for pid, m in matches.items())
            identity_conf = max(0.95 if (matched_prev or pb is not None) else 0.6, pb.identity_confidence if pb is not None else 0.0)
            beliefs[eid] = BeliefState(
                entity_id=eid,
                position_variance=0.05,
                velocity_variance=0.05,
                existence_probability=1.0,
                identity_confidence=min(identity_conf, 1.0),
                visibility_confidence=1.0,
                prediction_confidence=pred_conf,
            )

        # Entities that disappeared from view: decay existence, grow variance
        for pid, pent in prev_entities.items():
            if pid in merged or matches.get(pid) is not None:
                continue
            pb = prev_beliefs.get(pid)
            if pb is None:
                continue
            existence = pb.existence_probability * self.existence_decay
            if existence < self.drop_threshold:
                continue
            pos = np.asarray(pent.position, dtype=np.float32) + np.asarray(pent.velocity, dtype=np.float32)
            merged[pid] = EntityState(
                entity_id=pid,
                semantic_type=pent.semantic_type,
                position=pos.astype(np.float32),
                velocity=np.asarray(pent.velocity, dtype=np.float32).copy(),
                size=pent.size,
                appearance=pent.appearance,
                semantic_features=pent.semantic_features,
                dynamics_features=pent.dynamics_features,
                status=pent.status,
            )
            beliefs[pid] = BeliefState(
                entity_id=pid,
                position_variance=min(pb.position_variance + self.variance_growth, 2.0),
                velocity_variance=min(pb.velocity_variance + self.variance_growth, 2.0),
                existence_probability=existence,
                identity_confidence=pb.identity_confidence,
                visibility_confidence=0.0,
                prediction_confidence=pb.prediction_confidence,
            )

        entity_list = list(merged.values())
        belief_list = [beliefs[e.entity_id] for e in entity_list]
        spatial_arr = self.schema.spatial.from_raw(observation.spatial)
        spatial = SpatialMemory(data=spatial_arr, channels=list(self.schema.spatial.MEMORY_CHANNELS))
        return build_world_state(
            self.schema,
            entity_list,
            beliefs=belief_list,
            spatial=spatial,
            global_features=observation.global_features,
            t=t if t else observation.t,
            device=self.device,
            metadata={"tracker": "BeliefTracker", "tracked_ids": len(entity_list), "grid_size": int(observation.spatial.shape[1])},
        )

    def _fresh(self, observation: Observation, t: int) -> WorldState:
        beliefs = [
            BeliefState(entity_id=e.entity_id, position_variance=0.05, velocity_variance=0.05, existence_probability=1.0, identity_confidence=0.95, visibility_confidence=1.0, prediction_confidence=0.5)
            for e in observation.entities
        ]
        spatial_arr = self.schema.spatial.from_raw(observation.spatial)
        spatial = SpatialMemory(data=spatial_arr, channels=list(self.schema.spatial.MEMORY_CHANNELS))
        return build_world_state(
            self.schema,
            list(observation.entities),
            beliefs=beliefs,
            spatial=spatial,
            global_features=observation.global_features,
            t=t if t else observation.t,
            device=self.device,
            metadata={"tracker": "BeliefTracker", "grid_size": int(observation.spatial.shape[1])},
        )
