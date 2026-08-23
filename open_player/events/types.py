"""Event-specific types (the Event dataclass itself lives in core.types)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from open_player.core.types import Event, EventType

__all__ = ["Event", "EventType", "EventRelation"]


class EventRelation(str, Enum):
    """Edge kinds of the hierarchical event graph."""

    TEMPORAL = "temporal"
    CAUSAL = "causal"
    SPATIAL = "spatial"
