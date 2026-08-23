"""DummyVisionEncoder: structured observation -> WorldState (Phase 0).

A real vision encoder will later produce entity / spatial representations
from pixels; this dummy encoder consumes the synthetic environment's
structured observation and builds the same WorldState, so every downstream
module is already exercised against the final data flow.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from open_player.core.schema import SchemaSet
from open_player.core.specs import ObservationEncoder
from open_player.core.state import build_world_state
from open_player.core.types import BeliefState, Observation, SpatialMemory, WorldState


class DummyVisionEncoder(ObservationEncoder):
    """Structured observation -> single-batch WorldState."""

    def __init__(self, schema: SchemaSet, device: Any = "cpu") -> None:
        self.schema = schema
        self.device = device

    def encode(self, observation: Observation, t: int = 0) -> WorldState:
        entities = [e.copy() for e in observation.entities]
        beliefs = self._beliefs(observation)
        spatial_arr = self.schema.spatial.from_raw(observation.spatial)
        spatial = SpatialMemory(
            data=spatial_arr,
            channels=list(self.schema.spatial.MEMORY_CHANNELS),
            resolution=self.schema.spatial.H / max(observation.spatial.shape[1], 1),
        )
        return build_world_state(
            self.schema,
            entities,
            beliefs=beliefs,
            spatial=spatial,
            global_features=observation.global_features,
            t=t if t else observation.t,
            device=self.device,
            metadata={"encoder": "DummyVisionEncoder", "grid_size": int(observation.spatial.shape[1])},
        )

    def _beliefs(self, observation: Observation) -> List[BeliefState]:
        """Freshly-seen heuristic beliefs: visible entities are confident."""
        out: List[BeliefState] = []
        for e in observation.entities:
            out.append(
                BeliefState(
                    entity_id=e.entity_id,
                    position_variance=0.05,
                    velocity_variance=0.05,
                    existence_probability=1.0,
                    identity_confidence=0.95,
                    visibility_confidence=1.0,
                    prediction_confidence=0.5,
                )
            )
        return out
