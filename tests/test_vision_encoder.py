"""LearnedVisionEncoder tests: shapes, params, gradients, determinism."""
from __future__ import annotations

import numpy as np
import torch

from open_player.observation.dummy import DummyVisionEncoder
from open_player.observation.vision import LearnedVisionEncoder


def test_vision_params_under_budget(cfg_p1, schema_p1):
    enc = LearnedVisionEncoder(schema_p1, cfg_p1, device="cpu")
    n = enc.num_parameters()
    assert n < 3_000_000
    assert n > 100_000


def test_encode_shapes_and_rgb_path(cfg_p1, schema_p1, env_p1):
    enc = LearnedVisionEncoder(schema_p1, cfg_p1, device="cpu")
    obs = env_p1.reset(seed=5)
    assert obs.extra["rgb"].shape == (90, 160, 3)
    ws = enc.encode(obs, t=0)
    assert ws.entities_t.shape == (1, schema_p1.max_entities, 42)
    assert ws.spatial_t.shape == (1, 16, 32, 32)
    assert ws.metadata.get("encoder") == "LearnedVisionEncoder"
    # learned slices carry a computation graph
    assert ws.entities_t.requires_grad
    assert ws.spatial_t.requires_grad


def test_gradient_flow_to_conv(cfg_p1, schema_p1, env_p1):
    enc = LearnedVisionEncoder(schema_p1, cfg_p1, device="cpu")
    obs = env_p1.reset(seed=5)
    ws = enc.encode(obs, t=0)
    (ws.entities_t.sum() + ws.spatial_t.sum()).backward()
    # the encode path trains conv / spatial projection / patch mlp; the
    # occupancy + position heads receive their own auxiliary losses
    for name, module in (("conv", enc.conv), ("spatial_proj", enc.spatial_proj), ("patch_mlp", enc.patch_mlp)):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert len(grads) == len(list(module.parameters())), f"{name} missing gradients"


def test_determinism_same_encoder(cfg_p1, schema_p1, env_p1):
    enc = LearnedVisionEncoder(schema_p1, cfg_p1, device="cpu")
    obs = env_p1.reset(seed=5)
    ws1 = enc.encode(obs, t=0)
    ws2 = enc.encode(obs, t=0)
    assert torch.allclose(ws1.entities_t, ws2.entities_t)
    assert torch.allclose(ws1.spatial_t, ws2.spatial_t)


def test_dummy_encoder_still_works(cfg_p1, schema_p1, env_p1):
    """Phase 0 compatibility: the dummy path is untouched."""
    obs = env_p1.reset(seed=5)
    ws = DummyVisionEncoder(schema_p1).encode(obs)
    assert ws.entities_t.shape == (1, schema_p1.max_entities, 42)


def test_vision_needs_rgb(cfg_p1, schema_p1):
    from open_player.core.types import Observation
    enc = LearnedVisionEncoder(schema_p1, cfg_p1, device="cpu")
    obs = Observation(entities=[], spatial=np.zeros((8, 10, 10), dtype=np.float32), global_features=np.zeros(6, dtype=np.float32))
    try:
        enc.encode(obs, t=0)
        raised = False
    except ValueError:
        raised = True
    assert raised
