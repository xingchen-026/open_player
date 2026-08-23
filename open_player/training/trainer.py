"""WorldModelTrainer: the self-supervised world learning loop.

Every env step can produce an online update (transition -> loss -> backprop);
periodically a replay batch is trained as well.  The trainer owns the
optimizer, the replay buffer, the uncertainty signal and the checkpoints.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from open_player.core.config import Config
from open_player.core.schema import SchemaSet
from open_player.core.types import WorldState
from open_player.training.checkpoint import Checkpointer
from open_player.training.replay import ReplayBuffer
from open_player.world.model import WorldModel
from open_player.world.uncertainty import UncertaintyEstimator

log = logging.getLogger("open_player.training")


class WorldModelTrainer:
    """Owns optimizer / replay / uncertainty / checkpointing for the WorldModel."""

    def __init__(self, model: WorldModel, config: Config, schema: SchemaSet, device: Any = "cpu") -> None:
        self.model = model
        self.config = config
        self.schema = schema
        self.device = device
        tc = config.training
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(tc.learning_rate))
        self.replay = ReplayBuffer(capacity=int(tc.replay_capacity), seed=int(config.seed))
        self.uncertainty = UncertaintyEstimator(dim=schema.entity.D_entity)
        self.checkpointer = Checkpointer(
            directory=str(config.get("checkpoint.dir", "checkpoints")),
            keep_last=int(config.get("checkpoint.keep_last", 2)),
        )
        self.step = 0
        self.update_every = int(tc.update_every)
        self.replay_update_every = int(tc.replay_update_every)
        self.replay_updates_per_tick = int(tc.replay_updates_per_tick)
        self.batch_size = int(tc.batch_size)
        self.grad_clip = float(tc.grad_clip)
        self.history: List[Dict[str, float]] = []
        self.latest: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def online_step(
        self,
        state: WorldState,
        action: int,
        next_state: WorldState,
        reward: float,
        done: bool,
        change: float,
    ) -> Dict[str, float]:
        """Store the transition; optionally update on it."""
        self.replay.store(state, action, next_state, reward, done, change)
        metrics: Dict[str, float] = {}
        if self.step % self.update_every == 0:
            pred = self.model.predict(state, torch.tensor([action], device=self.device))
            with torch.no_grad():
                target_z = self.model.representation(next_state).z
            losses = self.model.loss(pred, next_state, change_label=torch.tensor([change], device=self.device), target_z=target_z)
            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            metrics = {k: float(v.detach().cpu()) for k, v in losses.items()}
            self.latest = dict(metrics)
            # uncertainty signal from the entity head error
            err = (pred.entities_pred.detach() - next_state.entities_t).mean(dim=(0, 1)).cpu().numpy()
            self.uncertainty.update(err)
        self.step += 1
        return metrics

    def replay_update(self) -> Dict[str, float]:
        """One batch update from the replay buffer."""
        batch = self.replay.sample(self.batch_size, self.schema, device=self.device)
        total, metrics = self.model.compute_batch_losses(
            state=batch["state"],
            actions=batch["action"],
            next_state=batch["next_state"],
            change_label=batch["change"],
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return metrics

    def tick(self) -> Optional[Dict[str, float]]:
        """Run replay updates when due; returns combined metrics or None."""
        if self.step <= 0 or self.step % self.replay_update_every != 0:
            return None
        if not self.replay.is_ready(self.batch_size):
            return None
        merged: Dict[str, float] = {}
        for _ in range(self.replay_updates_per_tick):
            for k, v in self.replay_update().items():
                merged[k] = merged.get(k, 0.0) + v / self.replay_updates_per_tick
        self.history.append(merged)
        self.latest = merged
        return merged

    def save_checkpoint(self, path: str, metrics: Optional[Dict[str, Any]] = None) -> str:
        extra = {
            "uncertainty": {k: v.tolist() for k, v in self.uncertainty.state_dict().items() if hasattr(v, "tolist")},
            "replay_len": len(self.replay),
        }
        return self.checkpointer.save(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            metrics=metrics or self.latest,
            config=self.config.to_dict(),
            extra=extra,
        )

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        meta = self.checkpointer.load(path, model=self.model, optimizer=self.optimizer, device=self.device)
        self.step = int(meta.get("step", 0))
        extra = meta.get("extra", {})
        if "uncertainty" in extra and "ema_sq" in extra["uncertainty"]:
            self.uncertainty.load_state_dict(extra["uncertainty"])
        return meta

    def metrics(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"step": self.step, "replay": self.replay.stats()}
        out.update(self.latest)
        out["uncertainty_mean"] = self.uncertainty.mean
        return out

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()
