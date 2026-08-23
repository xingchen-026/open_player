"""ObservationEncoder interface.

Phase 0 keeps the API that a real vision encoder will implement later, but
the default pipeline is: SyntheticEnvironment -> structured Observation ->
DummyVisionEncoder -> WorldState.
"""
from __future__ import annotations

from open_player.core.specs import ObservationEncoder

__all__ = ["ObservationEncoder"]
