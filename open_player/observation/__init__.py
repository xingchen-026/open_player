"""Observation layer: encoder interface + Phase 0 dummy vision encoder."""
from __future__ import annotations

from open_player.observation.dummy import DummyVisionEncoder
from open_player.observation.encoder import ObservationEncoder

__all__ = ["DummyVisionEncoder", "ObservationEncoder"]
