"""Hierarchical Event Graph: primitives -> composites -> episodes.

Every added event gets a temporal edge from the previous event (and a
parent_event link).  Causal and spatial edges are added explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from open_player.core.types import Event
from open_player.events.types import EventRelation


@dataclass
class EventEdge:
    src: str
    dst: str
    kind: str  # EventRelation value
    weight: float = 1.0


class EventGraph:
    """Stores events and temporal/causal/spatial relations between them."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Event] = {}
        self.edges: List[EventEdge] = []
        self.last_event_id: Optional[str] = None

    # -- events ---------------------------------------------------------- #
    def add_event(self, event: Event) -> Event:
        """Add an event; link it temporally to the previous one."""
        if event.event_id in self.nodes:
            return self.nodes[event.event_id]
        if self.last_event_id is not None and event.parent_event is None:
            event.parent_event = self.last_event_id
            self.edges.append(EventEdge(src=self.last_event_id, dst=event.event_id, kind=EventRelation.TEMPORAL.value, weight=1.0))
        self.nodes[event.event_id] = event
        self.last_event_id = event.event_id
        return event

    def add_events(self, events: List[Event]) -> List[Event]:
        return [self.add_event(e) for e in events]

    # -- relations ------------------------------------------------------- #
    def connect(self, src: str, dst: str, kind: str, weight: float = 1.0) -> None:
        if src in self.nodes and dst in self.nodes:
            self.edges.append(EventEdge(src=src, dst=dst, kind=kind, weight=weight))
        else:
            raise KeyError(f"cannot connect unknown events: {src} -> {dst}")

    def connect_causal(self, cause: str, effect: str, weight: float = 1.0) -> None:
        self.connect(cause, effect, EventRelation.CAUSAL.value, weight)

    def connect_spatial(self, a: str, b: str, weight: float = 1.0) -> None:
        self.connect(a, b, EventRelation.SPATIAL.value, weight)

    # -- queries --------------------------------------------------------- #
    def parents(self, event_id: str) -> List[Event]:
        return [self.nodes[e.src] for e in self.edges if e.dst == event_id]

    def children(self, event_id: str) -> List[Event]:
        return [self.nodes[e.dst] for e in self.edges if e.src == event_id]

    def sequence(self) -> List[Event]:
        """Events in insertion (temporal) order."""
        return list(self.nodes.values())

    def by_type(self, etype: str) -> List[Event]:
        return [e for e in self.nodes.values() if e.type == etype]

    def compose(self, event_ids: List[str], composite_type: str = "composite", t: Optional[int] = None) -> Event:
        """Create a composite event parenting the given primitives."""
        members = [self.nodes[eid] for eid in event_ids if eid in self.nodes]
        if not members:
            raise ValueError("compose needs at least one known event id")
        ts = t if t is not None else max(e.timestamp for e in members)
        comp = Event(
            event_id=f"composite-{len(self.nodes)}",
            type=composite_type,
            timestamp=int(ts),
            entities=sorted({eid for e in members for eid in e.entities}),
            location=members[-1].location,
            confidence=min(e.confidence for e in members),
            metadata={"members": event_ids},
        )
        self.add_event(comp)
        for eid in event_ids:
            if eid in self.nodes:
                self.edges.append(EventEdge(src=eid, dst=comp.event_id, kind=EventRelation.CAUSAL.value, weight=1.0))
        return comp

    # -- misc ------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        from collections import Counter
        return {
            "num_events": len(self.nodes),
            "num_edges": len(self.edges),
            "type_counts": dict(Counter(e.type for e in self.nodes.values())),
        }

    def reset(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.last_event_id = None
