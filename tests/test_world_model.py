"""World model tests: predict / rollout / loss / update."""
from __future__ import annotations

import torch


def test_predict_shapes(model, state_pair):
    s0, s1 = state_pair
    pred = model.predict(s0, 3)
    assert pred.entities_pred.shape == s1.entities_t.shape
    assert pred.spatial_pred.shape == s1.spatial_t.shape
    assert pred.change_logits.shape == (1, 2)
    assert pred.z_next.shape == (1, model.latent_dim)


def test_rollout_length(model, state_pair):
    preds = model.rollout(state_pair[0], [1, 2, 3, 4], k=8)
    assert len(preds) == 8
    assert all(p.entities_pred.shape == state_pair[0].entities_t.shape for p in preds)


def test_loss_keys(model, state_pair):
    s0, s1 = state_pair
    pred = model.predict(s0, 2)
    with torch.no_grad():
        tz = model.representation(s1).z
    losses = model.loss(pred, s1, change_label=torch.tensor([1.0]), target_z=tz)
    for key in ("entity", "spatial", "change", "latent", "total"):
        assert key in losses


def test_update_reduces_loss(model, state_pair):
    s0, s1 = state_pair
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(12):
        metrics = model.update(s0, 3, s1, opt, change_label=0.0, grad_clip=1.0)
        first = first if first is not None else metrics["total"]
    assert metrics["total"] < first


def test_param_budget(model):
    n = model.num_parameters()
    assert n < 10_000_000
