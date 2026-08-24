"""Observation layer: encoder interface, Phase 0 dummy encoder, Phase 1 vision."""
from __future__ import annotations

from open_player.observation.dummy import DummyVisionEncoder
from open_player.observation.encoder import ObservationEncoder
from open_player.observation.vision import LearnedVisionEncoder

__all__ = ["DummyVisionEncoder", "LearnedVisionEncoder", "ObservationEncoder"]
