"""LearnedVisionEncoder: RGB -> spatial/entity representation -> WorldState.

Phase 1 vision module (lightweight CNN, no transformer).  Phase 1.5 adds two
auxiliary-supervised heads used by the strict / learned_grid modes:

* occupancy head:  RGB -> wall/occupancy probability map (learned geometry)
* localization head: RGB -> the player's own position (learned self-localisation)

Both heads are trained ONLY on the training world (auxiliary losses); at
evaluation time the agent reads the learned estimates, never the ground
truth.  Modes:

* side         (Phase 1 default): GT struct_grid side-channel available
* learned_grid : wall channel = learned occupancy; novelty = visit-derived
* strict       : learned_grid + no GT entity positions + no env GT info
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from open_player.core.schema import SchemaSet
from open_player.core.specs import ObservationEncoder
from open_player.core.state import compute_relations, world_state_from_tensors
from open_player.core.types import BeliefState, Observation, WorldState

VISION_MODES = ("structured", "side", "learned_grid", "strict")


def _default_beliefs(entities: List[Any]) -> List[BeliefState]:
    out: List[BeliefState] = []
    for e in entities:
        out.append(BeliefState(
            entity_id=e.entity_id,
            position_variance=0.05,
            velocity_variance=0.05,
            existence_probability=1.0,
            identity_confidence=0.95,
            visibility_confidence=1.0,
            prediction_confidence=0.5,
        ))
    return out


class LearnedVisionEncoder(nn.Module, ObservationEncoder):
    """RGB -> WorldState (learned spatial + entity + occupancy + position)."""

    def __init__(self, schema: SchemaSet, config: Any, device: Any = "cpu") -> None:
        nn.Module.__init__(self)
        self.schema = schema
        self.config = config
        self.device = device
        vc = config.vision
        self.input_h = int(vc.get("input_height", 90))
        self.input_w = int(vc.get("input_width", 160))
        self.mode = str(vc.get("mode", "side"))
        if self.mode not in VISION_MODES:
            raise ValueError(f"unknown vision mode '{self.mode}' (expected one of {VISION_MODES})")
        channels = [int(c) for c in vc.channels]
        spatial_channels = int(vc.spatial_channels)
        patch_hidden = int(vc.patch_hidden)
        patch_out = int(vc.patch_out)

        layers: List[nn.Module] = []
        in_ch = 3
        for ch in channels:
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_ch = ch
        layers.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        self.feat_channels = in_ch

        self.spatial_proj = nn.Conv2d(in_ch, spatial_channels, kernel_size=1)
        self.spatial_norm = nn.InstanceNorm2d(spatial_channels, affine=True)
        self.patch_mlp = nn.Sequential(
            nn.Linear(in_ch, patch_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(patch_hidden, patch_out),
        )

        # Phase 1.5 auxiliary heads (small; supervised on the training world)
        self.occupancy_head: Optional[nn.Module] = None
        if bool(vc.get("occupancy_head", True)):
            self.occupancy_head = nn.Sequential(
                nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
            )
        self.position_head = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )  # -> [B, 2] in [-1, 1] (u, v normalized coords)

    # ------------------------------------------------------------------ #
    def forward_rgb(self, rgb: Any) -> Tensor:
        if isinstance(rgb, np.ndarray):
            x = torch.from_numpy(np.asarray(rgb, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
        else:
            x = rgb
            if x.dim() == 4 and x.shape[1] != 3:
                x = x.permute(0, 3, 1, 2)
            elif x.dim() == 3 and x.shape[0] != 3:
                x = x.permute(2, 0, 1).unsqueeze(0)
        x = x.to(device=self.device, dtype=torch.float32) / 255.0
        return self.conv(x)

    def spatial_features(self, feat_map: Tensor) -> Tensor:
        sp = self.spatial_proj(feat_map)
        sp = self.spatial_norm(sp)
        sp = F.interpolate(sp, size=(self.schema.spatial.H, self.schema.spatial.W), mode="bilinear", align_corners=False)
        return sp

    def occupancy_logits(self, feat_map: Tensor) -> Optional[Tensor]:
        if self.occupancy_head is None:
            return None
        return self.occupancy_head(feat_map.detach())  # [B, 1, H', W']

    def position_estimate(self, feat_map: Tensor) -> Tensor:
        """Learned self-localisation: global-pooled features -> [B, 2] tanh (u, v)."""
        pooled = feat_map.mean(dim=(2, 3))
        return torch.tanh(self.position_head(pooled))

    def patch_features(self, feat_map: Tensor, entities: List[Any], grid_size: int) -> Tensor:
        if not entities:
            return feat_map.new_zeros((0, int(self.config.vision.patch_out)))
        B, C, H, W = feat_map.shape
        coords = []
        for e in entities:
            x = float(e.position[0])
            y = float(e.position[1])
            u = (x + 0.5) / float(grid_size) * 2.0 - 1.0
            v = (y + 0.5) / float(grid_size) * 2.0 - 1.0
            coords.append([u, v])
        grid = torch.tensor(coords, dtype=torch.float32, device=feat_map.device).view(1, -1, 1, 2)
        sampled = F.grid_sample(feat_map, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.permute(0, 2, 3, 1).reshape(-1, C)
        return self.patch_mlp(sampled)

    # ------------------------------------------------------------------ #
    def encode(self, observation: Observation, t: int = 0) -> WorldState:
        rgb = observation.extra.get("rgb")
        if rgb is None:
            raise ValueError("LearnedVisionEncoder needs observation.extra['rgb'] (render_rgb: true)")
        gs = int(observation.extra.get("grid_size", self.schema.world_size))
        feat_map = self.forward_rgb(rgb)
        spatial = self.spatial_features(feat_map)  # [1, 16, 32, 32]
        occ_logits = self.occupancy_logits(feat_map)
        pos_est = self.position_estimate(feat_map)  # [1, 2] in [-1,1]

        entities = list(observation.entities)
        n = len(entities)
        base = np.stack([self.schema.entity.encode(e) for e in entities]).astype(np.float32) if n else np.zeros((0, self.schema.entity.D_entity), dtype=np.float32)
        ent = torch.from_numpy(base).to(device=self.device, dtype=torch.float32).clone()

        vis = self.patch_features(feat_map, entities, gs)
        if vis.shape[0] > 0:
            a_s, a_e = self.schema.entity.field_slice("appearance")
            s_s, s_e = self.schema.entity.field_slice("semantic_features")
            ent[:, a_s:a_e] = vis[:, : a_e - a_s]
            ent[:, s_s:s_e] = vis[:, a_e - a_s : (a_e - a_s) + (s_e - s_s)]

        # strict: replace GT positions with the LEARNED self-localisation
        # (player slot only; other entities are not localised in Phase 1.5)
        if self.mode == "strict":
            p_s, p_e = self.schema.entity.field_slice("position")
            v_s, v_e = self.schema.entity.field_slice("velocity")
            learned_uv = pos_est[0].detach().cpu().numpy()  # [-1, 1]
            learned_xy = np.array([
                (learned_uv[0] + 1.0) / 2.0 * gs,
                (learned_uv[1] + 1.0) / 2.0 * gs,
            ], dtype=np.float32)
            ent[:, p_s:p_e] = 0.0
            ent[:, v_s:v_e] = 0.0
            for i, e in enumerate(entities):
                if e.semantic_type == "player":
                    ent[i, p_s:p_e] = torch.tensor(learned_xy / self.schema.entity.world_size, device=ent.device, dtype=ent.dtype)

        beliefs = _default_beliefs(entities)
        belief_vecs = np.stack([self.schema.belief.encode(b) for b in beliefs]).astype(np.float32) if n else np.zeros((0, self.schema.belief.dim), dtype=np.float32)
        bel = torch.from_numpy(belief_vecs).to(device=self.device, dtype=torch.float32)
        rel_np = compute_relations(self.schema, entities, beliefs) if self.mode != "strict" else np.zeros((n, n, self.schema.relation.R), dtype=np.float32)
        rel = torch.from_numpy(rel_np).to(device=self.device, dtype=torch.float32)

        D_g = self.schema.global_dim
        glob = np.zeros(D_g, dtype=np.float32)
        gf = np.asarray(observation.global_features, dtype=np.float32).reshape(-1)
        glob[: min(D_g, gf.size)] = gf[:D_g]
        global_t = torch.from_numpy(glob).to(device=self.device, dtype=torch.float32)

        # learned occupancy map (grid resolution, detached) for learned_grid/strict
        learned_grid: Dict[str, np.ndarray] = {}
        if occ_logits is not None and self.mode in ("learned_grid", "strict"):
            occ_prob = torch.sigmoid(occ_logits[0, 0]).detach().cpu().numpy()  # [H', W']
            learned_grid["wall"] = self._resize_map(occ_prob, gs)

        # structured side-channel only in 'side' mode
        struct_grid: Dict[str, np.ndarray] = {}
        if self.mode == "side":
            raw = list(observation.extra.get("raw_channels", []))
            obs_spatial = np.asarray(observation.spatial, dtype=np.float32)
            if raw:
                for idx, name in enumerate(raw):
                    struct_grid[name] = obs_spatial[idx]
                if "visited" in struct_grid:
                    struct_grid["novelty"] = 1.0 - struct_grid["visited"]

        metadata = {
            "encoder": "LearnedVisionEncoder",
            "grid_size": gs,
            "mode": self.mode,
            "strict_rgb": self.mode == "strict",
            "occ_logits": occ_logits,
            "pos_estimate": pos_est,
            "learned_grid": learned_grid,
            "struct_grid": struct_grid,
        }

        return world_state_from_tensors(
            self.schema,
            entity_ids=[e.entity_id for e in entities],
            semantic_types=[e.semantic_type for e in entities],
            entities_t=ent,
            beliefs_t=bel,
            relations_t=rel,
            spatial_t=spatial[0],
            global_t=global_t,
            t=t if t else observation.t,
            device=self.device,
            metadata=metadata,
        )

    @staticmethod
    def _resize_map(arr: np.ndarray, gs: int) -> np.ndarray:
        """Nearest-neighbour resize of an [H', W'] map to [gs, gs]."""
        h, w = arr.shape
        ys = np.clip(np.floor(np.linspace(0, h, gs)).astype(np.int64), 0, h - 1)
        xs = np.clip(np.floor(np.linspace(0, w, gs)).astype(np.int64), 0, w - 1)
        return arr[np.ix_(ys, xs)]

    def aux_losses(self, state: WorldState, observation: Observation) -> Optional[torch.Tensor]:
        """Auxiliary losses (occupancy BCE + position MSE) against GT.

        Only used DURING training on the training world; evaluation never
        calls this.  Returns None when no auxiliary head is active.
        """
        occ_logits = state.metadata.get("occ_logits")
        pos_est = state.metadata.get("pos_estimate")
        if occ_logits is None and pos_est is None:
            return None
        total = None
        if occ_logits is not None:
            raw = list(observation.extra.get("raw_channels", []))
            if "wall" in raw:
                idx = raw.index("wall")
                wall_grid = np.asarray(observation.spatial[idx], dtype=np.float32)
                h, w = occ_logits.shape[2], occ_logits.shape[3]
                ys = np.clip(np.floor(np.linspace(0, wall_grid.shape[0], h)).astype(np.int64), 0, wall_grid.shape[0] - 1)
                xs = np.clip(np.floor(np.linspace(0, wall_grid.shape[1], w)).astype(np.int64), 0, wall_grid.shape[1] - 1)
                target = torch.from_numpy(wall_grid[np.ix_(ys, xs)]).to(device=occ_logits.device, dtype=occ_logits.dtype).unsqueeze(0).unsqueeze(0)
                w_occ = float(self.config.get("vision.occupancy_loss_weight", 0.5))
                total = w_occ * F.binary_cross_entropy_with_logits(occ_logits, target)
        if pos_est is not None:
            gs = int(state.metadata.get("grid_size", self.schema.world_size))
            player = next((e for e in observation.entities if e.semantic_type == "player"), None)
            if player is not None:
                u = (float(player.position[0]) + 0.5) / gs * 2.0 - 1.0
                v = (float(player.position[1]) + 0.5) / gs * 2.0 - 1.0
                target = torch.tensor([[u, v]], device=pos_est.device, dtype=pos_est.dtype)
                w_pos = float(self.config.get("vision.position_loss_weight", 0.5))
                lpos = w_pos * F.mse_loss(pos_est, target)
                total = lpos if total is None else total + lpos
        return total

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
