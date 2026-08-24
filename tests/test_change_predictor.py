"""Learned change / boundary predictor + hybrid event tests."""
from __future__ import annotations

import numpy as np
import torch

from open_player.core.config import set_seed
from open_player.core.schema import SchemaSet
from open_player.core.types import Event, EventType
from open_player.environments.synthetic.env import SyntheticGridEnv
from open_player.events.detector import HeuristicEventDetector, HybridEventDetector
from open_player.tracking.tracker import BeliefTracker
from open_player.world.change import LearnedChangePredictor
from open_player.world.model import WorldModel


def test_predictor_shapes():
    p = LearnedChangePredictor(latent_dim=64, num_actions=6, hidden=64)
    z = torch.randn(4, 64)
    a = torch.randint(0, 6, (4,))
    logits, boundary = p(z, a, z + 0.1)
    assert logits.shape == (4, 2)
    assert boundary.shape == (4, 1)
    assert (boundary >= 0).all() and (boundary <= 1).all()
    assert p.num_parameters() < 1_000_000


def test_predictor_learns_separable_change():
    set_seed(0)
    p = LearnedChangePredictor(latent_dim=16, num_actions=6, hidden=32)
    opt = torch.optim.Adam(p.parameters(), lr=1e-2)
    z = torch.randn(256, 16)
    a = torch.randint(0, 6, (256,))
    small = z + 0.02 * torch.randn(256, 16)
    large = z + 1.0 * torch.randn(256, 16)
    z_t = torch.cat([z, z])
    a_all = torch.cat([a, a])
    z1 = torch.cat([small, large])
    labels = torch.cat([torch.zeros(256), torch.ones(256)])
    first = None
    for _ in range(60):
        logits, _ = p(z_t, a_all, z1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 1], labels)
        first = first if first is not None else float(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert float(loss) < first


def test_hybrid_detector_blends_confidence(cfg_p1, schema_p1):
    set_seed(int(cfg_p1.seed))
    env = SyntheticGridEnv(cfg_p1)
    tr = BeliefTracker(schema_p1)
    model = WorldModel(schema_p1, cfg_p1, num_actions=6)
    # push the predictor toward "no change" so blending is visible
    with torch.no_grad():
        model.change_predictor.net[-1].bias[1] = -5.0
        model.change_predictor.net[-1].bias[2] = -5.0
    obs0 = env.reset(seed=1)
    s0 = tr.track(None, obs0, 0)
    obs1, _, _, info = env.step(3)
    s1 = tr.track(s0, obs1, 1)
    det = HybridEventDetector(HeuristicEventDetector(), world_model=model, conf_blend=0.5)
    events = det.detect(s0, s1, {**info, "action": 3}, 1)
    assert events
    for e in events:
        assert "learned_change_prob" in e.metadata
        assert "boundary_prob" in e.metadata
        assert e.confidence < 1.0  # blended down by the pessimistic predictor
    # heuristic detector alone is untouched
    plain = HeuristicEventDetector().detect(s0, s1, info, 1)
    assert len(plain) == len(events)
