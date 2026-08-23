"""Checkpoint save/load with explicit versioning."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch

CHECKPOINT_KIND = "open-player-checkpoint"
CHECKPOINT_VERSION = 1


class Checkpointer:
    """Versioned torch checkpoint manager."""

    def __init__(self, directory: str = "checkpoints", keep_last: int = 2) -> None:
        self.directory = str(directory)
        self.keep_last = int(keep_last)

    def save(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "kind": CHECKPOINT_KIND,
            "version": CHECKPOINT_VERSION,
            "step": int(step),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model_state": model.state_dict(),
            "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "metrics": dict(metrics or {}),
            "config": dict(config or {}),
            "extra": dict(extra or {}),
        }
        torch.save(payload, path)
        self._prune(path)
        return path

    def load(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Any = "cpu",
    ) -> Dict[str, Any]:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("kind") != CHECKPOINT_KIND:
            raise ValueError(f"not an Open Player checkpoint: {path}")
        if int(payload.get("version", 0)) != CHECKPOINT_VERSION:
            raise ValueError(
                f"checkpoint version mismatch: file={payload.get('version')} expected={CHECKPOINT_VERSION}"
            )
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        return {
            "step": int(payload["step"]),
            "metrics": dict(payload.get("metrics", {})),
            "config": dict(payload.get("config", {})),
            "extra": dict(payload.get("extra", {})),
            "saved_at": payload.get("saved_at"),
            "version": int(payload["version"]),
        }

    def _prune(self, keep_path: str) -> None:
        """Keep only the newest keep_last checkpoints (best-effort)."""
        if self.keep_last <= 0:
            return
        d = os.path.dirname(os.path.abspath(keep_path))
        try:
            files = sorted(
                (os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pt")),
                key=os.path.getmtime,
                reverse=True,
            )
            for old in files[self.keep_last :]:
                try:
                    os.remove(old)
                except OSError:  # pragma: no cover
                    pass
        except OSError:  # pragma: no cover
            pass
