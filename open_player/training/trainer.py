"""WorldModelTrainer: the self-supervised world learning loop.

Phase 0 behaviour (1-step online update + replay batch) is unchanged when the
Phase 1 config sections are absent.  With the Phase 1 config, the trainer
additionally:

* trains the learned change / boundary predictor (event_pred.enabled);
* trains multi-step horizons (multi_step.horizons, e.g. 4 and 8) from a
  sliding sequence window, with a scheduled teacher-forcing curriculum
  (rollout_schedule: teacher forcing -> mixed rollout -> model rollout);
* keeps targets detached when the states come from a learned encoder
  (vision.detach_targets), so the encoder learns predictive features.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

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

        # -- Phase 1: multi-step + scheduled rollout -------------------- #
        self.ms_horizons: List[int] = [int(h) for h in config.get("multi_step.horizons", [])]
        self.ms_weights: Dict[str, float] = dict(config.get("multi_step.loss_weights", {}))
        self.ms_window = int(config.get("multi_step.window", 8))
        self.ms_enabled = len(self.ms_horizons) > 0
        self.tf_initial = float(config.get("rollout_schedule.initial_teacher_forcing", 1.0))
        self.tf_final = float(config.get("rollout_schedule.final_teacher_forcing", 1.0))
        self.tf_anneal_steps = int(config.get("rollout_schedule.anneal_steps", 5000))
        self.rollout_ratio = float(config.get("rollout_schedule.rollout_ratio", 0.0))
        self.detach_targets = bool(config.get("vision.detach_targets", False))
        self.event_pred_enabled = bool(config.get("event_pred.enabled", False))
        self._seq: Deque[Tuple[WorldState, int]] = deque(maxlen=self.ms_window + 1)
        self._rng = torch.Generator().manual_seed(int(config.seed))

    # ------------------------------------------------------------------ #
    def teacher_forcing_ratio(self) -> float:
        """Annealed teacher-forcing ratio (0.9 -> 0.2 by default)."""
        if self.tf_anneal_steps <= 0:
            return self.tf_final
        progress = min(1.0, self.step / float(self.tf_anneal_steps))
        return self.tf_initial + (self.tf_final - self.tf_initial) * progress

    def _multi_step_teacher_forcing(self) -> float:
        """Per-update teacher-forcing probability (scheduled rollout)."""
        r = torch.rand(1, generator=self._rng).item()
        if r < self.rollout_ratio:
            return 0.0  # pure model rollout
        return self.teacher_forcing_ratio()  # per-step tf probability (mixed)

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
        """Store the transition; optionally update on it (1-step + multi-step)."""
        self.replay.store(state, action, next_state, reward, done, change)
        metrics: Dict[str, float] = {}
        if self.step % self.update_every == 0:
            a_t = torch.tensor([action], device=self.device)
            pred = self.model.predict(state, a_t)
            with torch.no_grad():
                target_z = self.model.representation(next_state).z
            losses = self.model.loss(
                pred, next_state,
                change_label=torch.tensor([change], device=self.device),
                target_z=target_z,
                detach_targets=self.detach_targets,
            )
            total = losses["total"]

            # learned change / boundary predictor replaces the z-only head
            if self.event_pred_enabled and self.model.change_predictor is not None:
                logits, boundary = self.model.predict_change(state, a_t, next_state)
                label = torch.tensor([change], device=self.device)
                from open_player.training.losses import boundary_loss, learned_change_loss
                lc = learned_change_loss(logits, label)
                lb = boundary_loss(boundary, label)
                w_ep = float(self.config.get("event_pred.loss_weight", 1.0))
                w_old = float(self.model.loss_weights.get("change", 0.5))
                total = total - w_old * losses.get("change", 0.0) + w_ep * lc + w_ep * 0.5 * lb
                metrics["learned_change"] = float(lc.detach().cpu())
                metrics["boundary"] = float(lb.detach().cpu())

            # anti-collapse regularizer on learned spatial features
            if self.detach_targets and state.spatial_t.requires_grad:
                from open_player.training.losses import spatial_variance_loss
                reg_w = float(self.config.get("vision.spatial_reg", 0.05))
                reg = reg_w * spatial_variance_loss(state.spatial_t)
                total = total + reg
                metrics["spatial_reg"] = float(reg.detach().cpu())

            # multi-step loss from the sliding sequence window
            if self.ms_enabled:
                # store detached states: each state's graph is consumed by
                # its own 1-step update; multi-step teacher forcing only
                # re-encodes them
                self._seq.append((state.detach(), action))
                if len(self._seq) == self.ms_window + 1:
                    seq = list(self._seq)
                    tf = self._multi_step_teacher_forcing()
                    ms = self.model.multi_step_loss(
                        state=seq[0][0],
                        actions=[a for _, a in seq[1:]],
                        target_states=[s for s, _ in seq[1:]],
                        teacher_forcing=tf,
                        detach_targets=self.detach_targets,
                    )
                    total = total + ms["total_ms"]
                    for name, value in ms.items():
                        if name != "total_ms":
                            metrics[f"ms_{name}"] = float(value.detach().cpu())
                    metrics["teacher_forcing"] = tf

            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            metrics["total"] = float(total.detach().cpu())
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and k != "total":
                    metrics.setdefault(k, float(v.detach().cpu()))
            self.latest = dict(metrics)
            # uncertainty signal from the entity head error
            err = (pred.entities_pred.detach() - next_state.entities_t.detach()).mean(dim=(0, 1)).cpu().numpy()
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
        self.latest = {**self.latest, **merged}
        return merged

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str, metrics: Optional[Dict[str, Any]] = None, extra_modules: Optional[Dict[str, nn.Module]] = None) -> str:
        extra = {
            "uncertainty": {k: v.tolist() for k, v in self.uncertainty.state_dict().items() if hasattr(v, "tolist")},
            "replay_len": len(self.replay),
            "teacher_forcing_ratio": self.teacher_forcing_ratio(),
        }
        return self.checkpointer.save(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            metrics=metrics or self.latest,
            config=self.config.to_dict(),
            extra=extra,
            modules=extra_modules,
        )

    def load_checkpoint(self, path: str, modules: Optional[Dict[str, nn.Module]] = None) -> Dict[str, Any]:
        meta = self.checkpointer.load(path, model=self.model, optimizer=self.optimizer, device=self.device, modules=modules)
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
