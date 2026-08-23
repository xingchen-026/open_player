"""Intrinsic motivation: novelty / curiosity / progress drives (Phase 0)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from open_player.core.types import WorldState


class IntrinsicMotivation:
    """Computes intrinsic drive signals from the current WorldState.

    * novelty  - mean of the spatial novelty channel
    * curiosity - the world model's recent prediction error (scaled)
    * progress - EMA of recent reward magnitude
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._progress_ema = 0.0
        self._progress_decay = 0.9

    def compute(
        self,
        state: WorldState,
        world_model_error: Optional[float] = None,
        reward: float = 0.0,
    ) -> Dict[str, float]:
        sp = state.spatial_t[0].detach().cpu().numpy()
        novelty = float(sp[3].mean()) if sp.shape[0] > 3 else 1.0
        unknown = float(sp[6].mean()) if sp.shape[0] > 6 else 0.0
        threat = self._local_threat(state)
        curiosity = min(max(float(world_model_error or 0.0) / 10.0, 0.0), 1.0)
        self._progress_ema = self._progress_decay * self._progress_ema + (1.0 - self._progress_decay) * abs(float(reward))
        return {
            "novelty": float(novelty),
            "unknown": float(unknown),
            "threat": float(threat),
            "curiosity": float(curiosity),
            "progress": float(self._progress_ema),
        }

    def reset(self) -> None:
        self._progress_ema = 0.0

    @staticmethod
    def _local_threat(state: WorldState) -> float:
        """Threat level at the player's own grid cell (not the global max)."""
        try:
            from open_player.core.state import grid_channel
            player = next((e for e in state.entity_states(0) if e.semantic_type == "player"), None)
            if player is None:
                return 0.0
            threat = grid_channel(state, 2)
            gx = int(round(float(player.position[0])))
            gy = int(round(float(player.position[1])))
            if 0 <= gx < threat.shape[1] and 0 <= gy < threat.shape[0]:
                return float(threat[gy, gx])
        except Exception:  # pragma: no cover - defensive
            pass
        sp = state.spatial_t[0].detach().cpu().numpy()
        return float(sp[2].max()) if sp.shape[0] > 2 else 0.0
