"""WorldModel: 1-step prediction + multi-step rollout API.

predict(state, action)  -> Prediction(state_{t+1} features)
rollout(state, actions) -> [Prediction, ...]  (k-step latent rollout)
loss(prediction, target) -> dict of losses (+ total)
update(...)             -> one gradient step
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from open_player.core.schema import SchemaSet
from open_player.core.types import EntityState, WorldState
from open_player.world.dynamics import DynamicsModel
from open_player.world.representation import Representation, WorldRepresentation


@dataclass
class Prediction:
    """Model output for one (state, action) pair."""

    z_cur: Tensor  # [B, D_latent]
    z_next: Tensor  # [B, D_latent]
    entities_pred: Tensor  # [B, N, D_entity]
    spatial_pred: Tensor  # [B, C, H, W]
    change_logits: Tensor  # [B, 2]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: Any, dtype: Any = None) -> "Prediction":
        self.z_cur = self.z_cur.to(device=device, dtype=dtype)
        self.z_next = self.z_next.to(device=device, dtype=dtype)
        self.entities_pred = self.entities_pred.to(device=device, dtype=dtype)
        self.spatial_pred = self.spatial_pred.to(device=device, dtype=dtype)
        self.change_logits = self.change_logits.to(device=device, dtype=dtype)
        return self

    def detach(self) -> "Prediction":
        return Prediction(
            z_cur=self.z_cur.detach(),
            z_next=self.z_next.detach(),
            entities_pred=self.entities_pred.detach(),
            spatial_pred=self.spatial_pred.detach(),
            change_logits=self.change_logits.detach(),
            meta=dict(self.meta),
        )

    def predicted_entity_states(self, schema: SchemaSet, batch: int = 0) -> List[EntityState]:
        """Decode the predicted entity tensor for one batch item."""
        vecs = self.entities_pred[batch].detach().cpu().numpy()
        out: List[EntityState] = []
        for i, vec in enumerate(vecs):
            out.append(schema.entity.decode(vec, entity_id=f"pred-{i}"))
        return out


class WorldModel(nn.Module):
    """Representation + multi-step dynamics + event/change prediction."""

    def __init__(self, schema: SchemaSet, config: Any, num_actions: int) -> None:
        super().__init__()
        self.schema = schema
        self.config = config
        self.num_actions = int(num_actions)
        wm = config.world_model
        self.latent_dim = int(wm.latent_dim)
        self.loss_weights: Dict[str, float] = dict(wm.loss_weights)

        self.representation = WorldRepresentation(schema, config)
        self.dynamics = DynamicsModel(
            num_actions=num_actions,
            latent_dim=self.latent_dim,
            hidden=int(wm.dynamics_hidden),
            action_embedding_dim=int(wm.action_embedding_dim),
        )
        head_hidden = int(wm.head_hidden)
        N, D_entity = schema.max_entities, schema.entity.D_entity
        C, H, W = schema.spatial.shape

        self.entity_head = nn.Sequential(
            nn.Linear(self.latent_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, N * D_entity),
        )
        self.spatial_proj = nn.Linear(self.latent_dim, 64 * 8 * 8)
        self.spatial_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, C, kernel_size=4, stride=2, padding=1),
        )
        self.change_head = nn.Linear(self.latent_dim, 2)

        # Phase 1: learned change / boundary predictor (opt-in, config-driven)
        self.change_predictor = None
        if bool(config.get("event_pred.enabled", False)):
            from open_player.world.change import LearnedChangePredictor
            self.change_predictor = LearnedChangePredictor(
                latent_dim=self.latent_dim,
                num_actions=num_actions,
                hidden=int(config.get("event_pred.hidden", 64)),
            )

    # ------------------------------------------------------------------ #
    # Prediction API
    # ------------------------------------------------------------------ #
    def encode(self, state: WorldState) -> Representation:
        return self.representation(state)

    def predict(self, state: WorldState, action: Tensor | int) -> Prediction:
        """1-step prediction: WorldState_t + Action_t -> predicted t+1 features."""
        z = self.representation(state).z
        a = self._action_tensor(action, z.shape[0], z.device)
        return self._predict_from_z(z, a)

    def _predict_from_z(self, z: Tensor, a: Tensor) -> Prediction:
        z_next = self.dynamics(z, a)
        N, D_entity = self.schema.max_entities, self.schema.entity.D_entity
        C, H, W = self.schema.spatial.shape
        entities = self.entity_head(z_next).view(z.shape[0], N, D_entity)
        sp = self.spatial_proj(z_next).view(z.shape[0], 64, 8, 8)
        spatial = self.spatial_head(sp)[:, :, :H, :W]
        change_logits = self.change_head(z_next)
        return Prediction(
            z_cur=z,
            z_next=z_next,
            entities_pred=entities,
            spatial_pred=spatial,
            change_logits=change_logits,
            meta={"action": a.detach().cpu().tolist()},
        )

    def rollout(self, state: WorldState, actions: Sequence[int], k: Optional[int] = None) -> List[Prediction]:
        """Multi-step latent rollout; returns k predictions (k = len(actions))."""
        k = len(actions) if k is None else int(k)
        if k <= 0:
            return []
        z = self.representation(state).z
        preds: List[Prediction] = []
        for i in range(k):
            a_idx = actions[i] if i < len(actions) else actions[-1]
            a = self._action_tensor(a_idx, z.shape[0], z.device)
            pred = self._predict_from_z(z, a)
            preds.append(pred)
            z = pred.z_next
        return preds

    def forward(self, state: WorldState, action: Tensor | int) -> Prediction:
        return self.predict(state, action)

    # ------------------------------------------------------------------ #
    # Loss / update API
    # ------------------------------------------------------------------ #
    def loss(self, prediction: Prediction, target_state: WorldState, change_label: Optional[Tensor] = None, target_z: Optional[Tensor] = None, detach_targets: bool = False) -> Dict[str, Tensor]:
        """Unweighted per-term losses plus the weighted total."""
        from open_player.training.losses import prediction_losses

        return prediction_losses(
            prediction=prediction,
            target_state=target_state,
            change_label=change_label,
            target_z=target_z,
            weights=self.loss_weights,
            detach_targets=detach_targets,
        )

    def update(
        self,
        state: WorldState,
        action: Tensor | int,
        next_state: WorldState,
        optimizer: torch.optim.Optimizer,
        change_label: Optional[float] = None,
        grad_clip: Optional[float] = None,
    ) -> Dict[str, float]:
        """One self-supervised gradient step on a single transition."""
        pred = self.predict(state, action)
        with torch.no_grad():
            target_z = self.representation(next_state).z
        label = None
        if change_label is not None:
            label = torch.tensor([float(change_label)], device=pred.z_cur.device)
        losses = self.loss(pred, next_state, change_label=label, target_z=target_z)
        total = losses["total"]
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(self.parameters(), float(grad_clip))
        optimizer.step()
        return {k: float(v.detach().cpu()) for k, v in losses.items() if isinstance(v, Tensor)}

    def compute_batch_losses(
        self,
        state: WorldState,
        actions: Tensor,
        next_state: WorldState,
        change_label: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Losses for a batched transition (used by the trainer on replay)."""
        pred = self.predict(state, actions)
        with torch.no_grad():
            target_z = self.representation(next_state).z
        losses = self.loss(pred, next_state, change_label=change_label, target_z=target_z)
        metrics = {k: float(v.detach().cpu()) for k, v in losses.items() if isinstance(v, Tensor)}
        return losses["total"], metrics

    # ------------------------------------------------------------------ #
    # Phase 1: multi-step training + evaluation
    # ------------------------------------------------------------------ #
    def multi_step_loss(
        self,
        state: WorldState,
        actions: Sequence[int],
        target_states: List[WorldState],
        teacher_forcing: float = 1.0,
        detach_targets: bool = True,
    ) -> Dict[str, Tensor]:
        """Multi-step losses at the configured horizons (step4 / step8).

        teacher_forcing in [0, 1]: 1.0 = pure teacher forcing, 0.0 = pure
        model rollout, in between = per-step teacher-forcing probability
        (mixed rollout, the scheduled-rollout curriculum).
        """
        from open_player.training.losses import entity_prediction_loss, latent_consistency_loss, spatial_prediction_loss

        horizons = [int(h) for h in self.config.get("multi_step.horizons", [])]
        if not horizons:
            return {"total_ms": torch.zeros((), device=state.entities_t.device)}
        max_h = max(horizons)
        weights = dict(self.config.get("multi_step.loss_weights", {}))
        w = dict(self.loss_weights)
        device = state.entities_t.device
        z_cur = self.representation(state).z
        terms: Dict[str, Tensor] = {}
        for k in range(1, max_h + 1):
            if k > len(target_states):
                break
            a = torch.full((z_cur.shape[0],), int(actions[k - 1]), dtype=torch.long, device=device)
            if k > 1 and float(teacher_forcing) > 0.0:
                use_tf = float(teacher_forcing) >= 1.0 or torch.rand(1, device=device).item() < float(teacher_forcing)
                if use_tf:
                    z_cur = self.representation(target_states[k - 2]).z.detach()
            pred = self._predict_from_z(z_cur, a)
            z_cur = pred.z_next
            if k in horizons:
                tgt = target_states[k - 1]
                term = (
                    float(w.get("entity", 1.0)) * entity_prediction_loss(pred, tgt, detach_target=detach_targets)
                    + float(w.get("spatial", 0.1)) * spatial_prediction_loss(pred, tgt, detach_target=detach_targets)
                    + float(w.get("latent", 0.1)) * latent_consistency_loss(pred, self.representation(tgt).z.detach())
                )
                terms[f"step{k}"] = term
        total = torch.zeros((), device=device)
        for name, term in terms.items():
            total = total + float(weights.get(name, 0.0)) * term
        terms["total_ms"] = total
        return terms

    @torch.no_grad()
    def prediction_errors(
        self,
        state: WorldState,
        actions: Sequence[int],
        target_states: List[WorldState],
        horizons: Sequence[int] = (1, 4, 8),
    ) -> Dict[str, float]:
        """Model-rollout prediction errors per horizon (Phase 1 benchmark).

        Reports entity / spatial / latent errors for each horizon, the
        teacher-forcing upper bound for long horizons, and the learned change
        probability on the first transition.
        """
        device = state.entities_t.device
        max_h = max(horizons)
        errors: Dict[str, float] = {}
        z_cur = self.representation(state).z
        for k in range(1, max_h + 1):
            if k > len(target_states):
                break
            a = torch.full((z_cur.shape[0],), int(actions[k - 1]), dtype=torch.long, device=device)
            pred = self._predict_from_z(z_cur, a)
            z_cur = pred.z_next
            if k in horizons:
                tgt = target_states[k - 1]
                errors[f"step{k}_entity"] = float(torch.nn.functional.mse_loss(pred.entities_pred, tgt.entities_t))
                errors[f"step{k}_spatial"] = float(torch.nn.functional.mse_loss(pred.spatial_pred, tgt.spatial_t))
                errors[f"step{k}_latent"] = float(torch.nn.functional.mse_loss(pred.z_next, self.representation(tgt).z))
        for k in horizons:
            if k <= 1 or k > len(target_states):
                continue
            z_tf = self.representation(target_states[k - 2]).z
            a = torch.full((z_tf.shape[0],), int(actions[k - 1]), dtype=torch.long, device=device)
            pred = self._predict_from_z(z_tf, a)
            errors[f"step{k}_entity_tf"] = float(torch.nn.functional.mse_loss(pred.entities_pred, target_states[k - 1].entities_t))
        if self.change_predictor is not None and len(target_states) >= 1:
            z0 = self.representation(state).z
            z1 = self.representation(target_states[0]).z
            a0 = torch.full((z0.shape[0],), int(actions[0]), dtype=torch.long, device=device)
            logits, boundary = self.change_predictor(z0, a0, z1)
            errors["change_prob"] = float(torch.sigmoid(logits[:, 1]).mean())
            errors["boundary_prob"] = float(boundary.mean())
        return errors

    def predict_change(self, state: WorldState, action: Tensor | int, next_state: WorldState) -> Tuple[Tensor, Tensor]:
        """(change_logits, boundary_score) from the learned predictor."""
        if self.change_predictor is None:
            raise RuntimeError("learned change predictor is disabled (event_pred.enabled=false)")
        z_t = self.representation(state).z
        z_t1 = self.representation(next_state).z
        a = self._action_tensor(action, z_t.shape[0], z_t.device)
        return self.change_predictor(z_t, a, z_t1)

    # ------------------------------------------------------------------ #
    def _action_tensor(self, action: Tensor | int, batch: int, device: Any) -> Tensor:
        if isinstance(action, Tensor):
            return action.to(device=device)
        return torch.full((batch,), int(action), dtype=torch.long, device=device)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
