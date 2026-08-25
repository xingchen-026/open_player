"""NeuralSkill: a real learned skill (MLP policy + learned termination).

WorldState -> small MLP -> action logits + termination logit.  No
transformer, < 1M parameters.  The skill obeys the frozen Skill interface:
initiation_condition, policy (act), termination_condition, outcome_model
(predict_outcome), memory (self.memory dict) and metadata; it is an option
with a learned termination signal, NOT a fixed action sequence.

Training is behavior cloning from successful rule trajectories
(training/skill_trainer.py), optionally followed by intrinsic-reward
fine-tuning later.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from open_player.core.schema import SchemaSet
from open_player.core.types import Action, WorldState
from open_player.skills.base import OutcomePrediction, Skill


class StateFeaturizer:
    """Fixed-size feature vector for skill policy inputs.

    existence-weighted entity pooling + global + spatial (mean/max per
    channel) + uncertainty + GRID-RESOLUTION novelty/wall/threat maps + a
    player-position one-hot.  The grid maps are what make the policy's
    spatial decisions learnable (an MLP needs the layout, not just global
    statistics).  Still < 1M parameters for the downstream skill.
    """

    def __init__(self, schema: SchemaSet, device: Any = "cpu") -> None:
        self.schema = schema
        self.device = device

    @property
    def dim(self) -> int:
        C = self.schema.spatial.C
        gs = self.schema.world_size
        return self.schema.entity.D_entity + self.schema.global_dim + 2 * C + self.schema.uncertainty_dim + 3 * gs * gs + gs * gs

    def features(self, state: WorldState) -> Tensor:
        from open_player.core.state import structured_grid
        ent = state.entities_t
        w = state.beliefs_t[:, :, 4:5].detach().clamp(min=0.0)
        pooled = (ent * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        sp = state.spatial_t
        sp_mean = sp.mean(dim=(2, 3))
        sp_max = sp.amax(dim=(2, 3))
        # maps are padded to the schema's nominal grid size so the feature
        # dimension is fixed for a given schema (envs may use smaller grids)
        gs = int(state.metadata.get("grid_size", self.schema.world_size))
        ws = int(self.schema.world_size)
        novel = self._pad_map(structured_grid(state, "novelty"), ws)
        wall = self._pad_map(structured_grid(state, "wall"), ws)
        threat = self._pad_map(structured_grid(state, "threat"), ws)
        player = next((e for e in state.entity_states(0) if e.semantic_type == "player"), None)
        onehot = np.zeros((ws, ws), dtype=np.float32)
        if player is not None:
            px = int(np.clip(round(float(player.position[0])), 0, gs - 1))
            py = int(np.clip(round(float(player.position[1])), 0, gs - 1))
            onehot[py, px] = 1.0
        novel_t = torch.from_numpy(novel.reshape(-1).astype(np.float32)).to(self.device)
        wall_t = torch.from_numpy(wall.reshape(-1).astype(np.float32)).to(self.device)
        threat_t = torch.from_numpy(threat.reshape(-1).astype(np.float32)).to(self.device)
        onehot_t = torch.from_numpy(onehot.reshape(-1)).to(self.device)
        maps = torch.cat([novel_t, wall_t, threat_t, onehot_t]).unsqueeze(0)
        pooled = pooled.to(self.device)
        global_t = state.global_t.to(self.device)
        sp_mean = sp_mean.to(self.device)
        sp_max = sp_max.to(self.device)
        unc = state.uncertainty_t.to(self.device)
        return torch.cat([pooled, global_t, sp_mean, sp_max, unc, maps], dim=-1).to(self.device).detach()

    @staticmethod
    def _pad_map(arr: np.ndarray, size: int) -> np.ndarray:
        """Zero-pad a grid map to (size, size) for a fixed feature dim."""
        out = np.zeros((size, size), dtype=np.float32)
        h = min(arr.shape[0], size)
        w = min(arr.shape[1], size)
        out[:h, :w] = arr[:h, :w]
        return out


class NeuralSkill(nn.Module, Skill):
    """Learned skill: MLP policy + learned termination (BC-trained)."""

    def __init__(
        self,
        name: str,
        action_names: List[str],
        featurizer: Optional[StateFeaturizer] = None,
        horizon: int = 8,
        hidden: int = 128,
        mode: str = "greedy",
        term_threshold: float = 0.7,
    ) -> None:
        nn.Module.__init__(self)
        Skill.__init__(self, name=name, horizon=horizon)
        self.action_names = list(action_names)
        self.num_actions = len(self.action_names)
        self.featurizer = featurizer
        self.mode = mode
        self.term_threshold = float(term_threshold)
        self.hidden = int(hidden)
        d_in = featurizer.dim if featurizer is not None else 128
        self.net = nn.Sequential(
            nn.Linear(d_in, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
        )
        self.action_head = nn.Linear(self.hidden, self.num_actions)
        self.termination_head = nn.Linear(self.hidden, 1)
        self._done = False
        self.metadata: dict = {"kind": "neural", "mode": mode}

    # -- Skill interface ------------------------------------------------- #
    def can_start(self, state: WorldState) -> bool:
        return True

    def act(self, state: WorldState, rng: Optional[np.random.Generator] = None) -> Action:
        if self.featurizer is None:
            raise RuntimeError("NeuralSkill needs a StateFeaturizer (set skill.featurizer)")
        feats = self.featurizer.features(state)
        logits, term_logit = self.forward(feats)
        self._steps += 1
        if torch.sigmoid(term_logit).item() > self.term_threshold:
            self._done = True
        # validity mask: the learned policy picks WHERE to go; blocked moves
        # (walls / borders) are masked so it cannot push against geometry
        mask = self._valid_move_mask(state).to(logits.device).unsqueeze(0)
        masked = logits.clone()
        masked[~mask] = float("-inf")
        if masked.isfinite().any():
            logits = masked
        if self.mode == "sample":
            dist = torch.distributions.Categorical(logits=logits)
            idx = int(dist.sample().item())
        else:
            idx = int(logits.argmax(dim=-1).item())
        idx = min(max(idx, 0), self.num_actions - 1)
        return Action(name=self.action_names[idx], index=idx)

    def _valid_move_mask(self, state: WorldState) -> torch.Tensor:
        """True for non-move actions and moves into free (non-wall) cells."""
        from open_player.core.state import structured_grid
        mask = torch.ones(self.num_actions, dtype=torch.bool)
        player = next((e for e in state.entity_states(0) if e.semantic_type == "player"), None)
        if player is None:
            return mask
        wall = structured_grid(state, "wall")
        gs = int(state.metadata.get("grid_size", wall.shape[0]))
        px, py = round(float(player.position[0])), round(float(player.position[1]))
        for name, (dx, dy) in (("left", (-1, 0)), ("right", (1, 0)), ("up", (0, -1)), ("down", (0, 1))):
            if name in self.action_names:
                i = self.action_names.index(name)
                nx, ny = int(px + dx), int(py + dy)
                free = 0 <= nx < gs and 0 <= ny < gs and wall[ny, nx] <= 0.5
                mask[i] = bool(free)
        return mask

    def should_terminate(self, state: WorldState) -> bool:
        return self._done or self._steps >= self.horizon

    def predict_outcome(self, state: WorldState) -> OutcomePrediction:
        novelty = float(state.spatial_t[0, 3].detach().cpu().numpy().mean())
        return OutcomePrediction(self.name, expected_utility=0.5 * novelty + 0.3, expected_events=["move"], confidence=0.5)

    def update(self, *, state: Optional[WorldState] = None, action: Optional[Action] = None, reward: float = 0.0, next_state: Optional[WorldState] = None, done: bool = False, **kwargs: Any) -> None:
        """Hook for future RL fine-tuning (Phase 1 trains via BC)."""

    def reset(self) -> None:
        Skill.reset(self)
        self._done = False

    # -- network --------------------------------------------------------- #
    def forward(self, feats: Tensor) -> tuple:
        h = self.net(feats)
        return self.action_head(h), self.termination_head(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:  # pragma: no cover
        return f"NeuralSkill(name={self.name!r}, params={self.num_parameters()})"
