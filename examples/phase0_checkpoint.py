"""Checkpoint example: save -> load -> verify (versioned roundtrip).

Usage:
    python examples/phase0_checkpoint.py
    python examples/phase0_checkpoint.py --path checkpoints/phase0_example.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from open_player.core.config import default_config, set_seed
from open_player.core.schema import SchemaSet
from open_player.environments.synthetic.env import SyntheticGridEnv
from open_player.tracking.tracker import BeliefTracker
from open_player.training.trainer import WorldModelTrainer
from open_player.world.model import WorldModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Player checkpoint roundtrip example")
    parser.add_argument("--path", default="checkpoints/phase0_example.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0"])
    args = parser.parse_args()

    cfg = default_config()
    cfg.device = args.device
    set_seed(int(cfg.seed))
    from open_player.core.config import resolve_device
    device = resolve_device(cfg)

    schema = SchemaSet.from_config(cfg)
    env = SyntheticGridEnv(cfg)
    tracker = BeliefTracker(schema, device=device)
    obs0 = env.reset(seed=0)
    s0 = tracker.track(None, obs0, 0)
    obs1, _, _, _ = env.step(3)
    s1 = tracker.track(s0, obs1, 1)

    model = WorldModel(schema, cfg, num_actions=env.action_space.n).to(device)
    trainer = WorldModelTrainer(model, cfg, schema, device=device)
    trainer.online_step(s0, 3, s1, 0.0, False, 0.0)
    trainer.online_step(s1, 4, s1, 0.0, False, 0.0)
    print(f"[checkpoint] trained {trainer.step} steps; loss={trainer.latest}")

    trainer.save_checkpoint(args.path, metrics=trainer.latest)
    print(f"[checkpoint] saved -> {args.path}")

    model2 = WorldModel(schema, cfg, num_actions=env.action_space.n).to(device)
    trainer2 = WorldModelTrainer(model2, cfg, schema, device=device)
    meta = trainer2.load_checkpoint(args.path)
    print(f"[checkpoint] loaded -> step={meta['step']} version={meta['version']}")

    same = all(torch.allclose(p1, p2) for p1, p2 in zip(model.parameters(), model2.parameters()))
    print(f"[checkpoint] parameter roundtrip: {'OK (identical)' if same else 'FAILED'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
