"""LearnedVisionEncoder: RGB -> spatial/entity representation -> WorldState.

Phase 1 vision module.  A lightweight CNN (no transformer / ViT / pretrained
backbone) consumes the synthetic world's RGB frames (160x90) and produces:

* a learned spatial feature map, projected into the WorldState's spatial
  memory channels [16, 32, 32];
* learned appearance + semantic features for each entity, extracted from the
  feature map at the entity's location (differentiable patch sampling).

Positions / velocities / sizes still come from the structured observation
(Phase 1 explicitly allows this: the encoder produces the visual
representation, the existing tracking / state construction handles the rest).
The encoder is trained end-to-end by the world model's prediction loss: it
learns features that are useful for predicting the future, which is the
Phase 1 definition of "learning from vision".
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from open_player.core.schema import SchemaSet
from open_player.core.specs import ObservationEncoder
from open_player.core.state import compute_relations, world_state_from_tensors
from open_player.core.types import BeliefState, Observation, WorldState


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
    """RGB -> WorldState (learned spatial + entity features)."""

    def __init__(self, schema: SchemaSet, config: Any, device: Any = "cpu") -> None:
        nn.Module.__init__(self)
        self.schema = schema
        self.config = config
        self.device = device
        vc = config.vision
        self.input_h = int(vc.get("input_height", 90))
        self.input_w = int(vc.get("input_width", 160))
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
        # one extra same-resolution layer for capacity (~0.6M params)
        layers.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        self.feat_channels = in_ch

        self.spatial_proj = nn.Conv2d(in_ch, spatial_channels, kernel_size=1)
        # instance norm keeps the learned spatial map non-degenerate
        # (unit variance per channel) so the prediction loss is meaningful
        self.spatial_norm = nn.InstanceNorm2d(spatial_channels, affine=True)
        self.patch_mlp = nn.Sequential(
            nn.Linear(in_ch, patch_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(patch_hidden, patch_out),
        )

    # ------------------------------------------------------------------ #
    # forward pieces
    # ------------------------------------------------------------------ #
    def forward_rgb(self, rgb: Any) -> Tensor:
        """RGB (uint8 [H,W,3] numpy or float tensor [B,3,H,W]) -> feature map."""
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
        """Project the feature map into the WorldState spatial channels."""
        sp = self.spatial_proj(feat_map)  # [B, 16, H', W']
        sp = self.spatial_norm(sp)
        sp = F.interpolate(sp, size=(self.schema.spatial.H, self.schema.spatial.W), mode="bilinear", align_corners=False)
        return sp

    def patch_features(self, feat_map: Tensor, entities: List[Any], grid_size: int) -> Tensor:
        """Learned appearance+semantic features per entity (patch sampling)."""
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
        # [B, C, N, 1] -> [N, C]
        sampled = sampled.permute(0, 2, 3, 1).reshape(-1, C)
        return self.patch_mlp(sampled)

    # ------------------------------------------------------------------ #
    # ObservationEncoder interface
    # ------------------------------------------------------------------ #
    def encode(self, observation: Observation, t: int = 0) -> WorldState:
        rgb = observation.extra.get("rgb")
        if rgb is None:
            raise ValueError("LearnedVisionEncoder needs observation.extra['rgb'] (render_rgb: true)")
        gs = int(observation.extra.get("grid_size", self.schema.world_size))
        feat_map = self.forward_rgb(rgb)
        spatial = self.spatial_features(feat_map)  # [1, 16, 32, 32]

        entities = list(observation.entities)
        n = len(entities)
        base = np.stack([self.schema.entity.encode(e) for e in entities]).astype(np.float32) if n else np.zeros((0, self.schema.entity.D_entity), dtype=np.float32)
        ent = torch.from_numpy(base).to(device=self.device, dtype=torch.float32).clone()

        vis = self.patch_features(feat_map, entities, gs)  # [N, patch_out]
        if vis.shape[0] > 0:
            a_s, a_e = self.schema.entity.field_slice("appearance")
            s_s, s_e = self.schema.entity.field_slice("semantic_features")
            ent[:, a_s:a_e] = vis[:, : a_e - a_s]
            ent[:, s_s:s_e] = vis[:, a_e - a_s : (a_e - a_s) + (s_e - s_s)]

        beliefs = _default_beliefs(entities)
        belief_vecs = np.stack([self.schema.belief.encode(b) for b in beliefs]).astype(np.float32) if n else np.zeros((0, self.schema.belief.dim), dtype=np.float32)
        bel = torch.from_numpy(belief_vecs).to(device=self.device, dtype=torch.float32)
        rel_np = compute_relations(self.schema, entities, beliefs)
        rel = torch.from_numpy(rel_np).to(device=self.device, dtype=torch.float32)

        D_g = self.schema.global_dim
        glob = np.zeros(D_g, dtype=np.float32)
        gf = np.asarray(observation.global_features, dtype=np.float32).reshape(-1)
        glob[: min(D_g, gf.size)] = gf[:D_g]
        global_t = torch.from_numpy(glob).to(device=self.device, dtype=torch.float32)

        # structured grid side-channel: the spatial memory tensor carries
        # LEARNED features; rule skills / intrinsic reward read the raw
        # environment channels from here (wall / threat / novelty / ...)
        struct_grid: dict = {}
        raw = list(observation.extra.get("raw_channels", []))
        obs_spatial = np.asarray(observation.spatial, dtype=np.float32)
        if raw:
            for idx, name in enumerate(raw):
                struct_grid[name] = obs_spatial[idx]
            if "visited" in struct_grid:
                struct_grid["novelty"] = 1.0 - struct_grid["visited"]

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
            metadata={"encoder": "LearnedVisionEncoder", "grid_size": gs, "struct_grid": struct_grid},
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
