"""Open Player - a low-compute, non-Transformer-first learning agent core for games.

Phase 0 delivers a minimal, testable, trainable, extensible Learning Agent
Core together with a synthetic grid world and a self-supervised world-model
training loop.
"""
from __future__ import annotations

__version__ = "0.1.0"

from open_player.core.config import Config, load_config, resolve_device, set_seed, setup_logging
from open_player.core.types import (
    Action,
    BeliefState,
    EntityState,
    Episode,
    Event,
    EventType,
    Goal,
    GoalType,
    Observation,
    Relation,
    SpatialMemory,
    WorldState,
)

__all__ = [
    "Action",
    "BeliefState",
    "Config",
    "EntityState",
    "Episode",
    "Event",
    "EventType",
    "Goal",
    "GoalType",
    "Observation",
    "Relation",
    "SpatialMemory",
    "WorldState",
    "__version__",
    "load_config",
    "resolve_device",
    "set_seed",
    "setup_logging",
]
