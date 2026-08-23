"""UncertaintyEstimator: cheap predictive-uncertainty signal.

Exponential moving average of per-dimension squared prediction errors.  It is
deliberately simple (no Bayesian NN in Phase 0) and feeds the planner's
confidence and the world model's temporal state.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


class UncertaintyEstimator:
    """EMA of squared prediction errors per dimension."""

    def __init__(self, dim: int, decay: float = 0.95) -> None:
        self.dim = int(dim)
        self.decay = float(decay)
        self._ema_sq = np.zeros(self.dim, dtype=np.float64)

    def update(self, error: np.ndarray) -> None:
        err = np.asarray(error, dtype=np.float64).reshape(-1)[: self.dim]
        self._ema_sq = self.decay * self._ema_sq + (1.0 - self.decay) * (err ** 2)

    def estimate(self) -> np.ndarray:
        """Per-dimension standard-deviation estimate."""
        return np.sqrt(self._ema_sq).astype(np.float32)

    @property
    def mean(self) -> float:
        return float(self.estimate().mean())

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"ema_sq": self._ema_sq.copy(), "dim": self.dim, "decay": self.decay}

    def load_state_dict(self, d: Dict) -> None:
        self._ema_sq = np.asarray(d["ema_sq"], dtype=np.float64).reshape(-1)[: self.dim].copy()
        self.decay = float(d.get("decay", self.decay))
