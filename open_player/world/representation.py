"""WorldRepresentation: WorldState -> latent representation.

Small, explicit neural modules only (convs + MLPs; no transformer, no VLM).
The representation keeps every input channel separate until the final fusion,
so each piece stays inspectable and replaceable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from open_player.core.schema import SchemaSet
from open_player.core.types import WorldState


@dataclass
class Representation:
    """Latent representation of one WorldState."""

    z: Tensor  # [B, D_latent]
    entity_emb: Tensor  # [B, D_entity_hidden] (pooled over entities)
    spatial_emb: Tensor  # [B, D_spatial_embed]

    def detach(self) -> "Representation":
        return Representation(z=self.z.detach(), entity_emb=self.entity_emb.detach(), spatial_emb=self.spatial_emb.detach())


class WorldRepresentation(nn.Module):
    """Maps a structured WorldState to a latent Representation."""

    def __init__(self, schema: SchemaSet, config: Any) -> None:
        super().__init__()
        wm = config.world_model
        self.schema = schema
        d_in_entity = schema.entity.D_entity + schema.belief.dim
        entity_hidden = int(wm.entity_hidden)
        spatial_channels = [int(c) for c in wm.spatial_channels]
        spatial_embed = int(wm.spatial_embed)
        relation_hidden = int(wm.relation_hidden)
        latent_dim = int(wm.latent_dim)

        # entity stream: shared per-entity MLP, pooled with mean
        self.entity_mlp = nn.Sequential(
            nn.Linear(d_in_entity, entity_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(entity_hidden, entity_hidden),
        )
        # spatial stream: two strided convs (32x32 -> 16x16 -> 8x8)
        c_in = schema.spatial.C
        layers: list[nn.Module] = []
        prev = c_in
        for ch in spatial_channels:
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            prev = ch
        self.spatial_conv = nn.Sequential(*layers)
        self.spatial_fc = nn.Linear(prev * 8 * 8, spatial_embed)

        # relation stream: pooled pairwise relations
        self.relation_fc = nn.Linear(schema.relation.R, relation_hidden)

        # global + dynamics/uncertainty stream
        self.global_fc = nn.Linear(schema.global_dim, 32)
        self.dyn_fc = nn.Linear(schema.dynamics_dim + schema.uncertainty_dim, 32)

        # fusion
        fused_in = entity_hidden + spatial_embed + relation_hidden + 32 + 32
        self.fuse = nn.Linear(fused_in, latent_dim)

    def forward(self, state: WorldState) -> Representation:
        B = state.batch_size
        # entities + beliefs
        ent = torch.cat([state.entities_t, state.beliefs_t], dim=-1)  # [B, N, Din]
        ent = self.entity_mlp(ent).mean(dim=1)  # [B, entity_hidden]

        # spatial
        sp = self.spatial_conv(state.spatial_t)  # [B, ch_last, 8, 8]
        sp = self.spatial_fc(sp.flatten(1))  # [B, spatial_embed]

        # relations (pooled over pairs)
        rel = state.relations_t.mean(dim=(1, 2))  # [B, R]
        rel = torch.relu(self.relation_fc(rel))

        # global + dynamics/uncertainty
        g = torch.relu(self.global_fc(state.global_t))
        du = torch.relu(self.dyn_fc(torch.cat([state.dynamics_t, state.uncertainty_t], dim=-1)))

        fused = torch.cat([ent, sp, rel, g, du], dim=-1)
        z = self.fuse(fused)
        return Representation(z=z, entity_emb=ent, spatial_emb=sp)

    @property
    def latent_dim(self) -> int:
        return int(self.fuse.out_features)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
