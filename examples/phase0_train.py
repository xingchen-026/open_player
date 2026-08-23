"""Phase 0 training example.

Runs the full closed loop:

    Synthetic World -> Observation -> WorldState -> WorldModel -> Prediction
    -> Loss -> Backprop -> Checkpoint
    (+ Events -> Episodes -> Goals -> Planner -> Skills -> Actions)

Usage:
    python examples/phase0_train.py --config configs/phase0.yaml
    python examples/phase0_train.py --steps 5000 --device auto
    python examples/phase0_train.py --checkpoint checkpoints/phase0.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from open_player.agent.player import Player
from open_player.core.config import load_config, resolve_device, set_seed, setup_logging
from open_player.environments.synthetic.env import SyntheticGridEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Player Phase 0 training example")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "cuda:0"])
    parser.add_argument("--checkpoint", default=None, help="save a checkpoint to this path")
    parser.add_argument("--no-verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    setup_logging(cfg.get("logging.level", "INFO"))
    set_seed(int(cfg.seed))

    device = resolve_device(cfg)
    print(f"[phase0_train] config={args.config} seed={cfg.seed} device={device}")
    print(f"[phase0_train] device note: {('CUDA' if str(device).startswith('cuda') else 'CPU')} "
          f"({'auto-detected' if args.device in (None, 'auto') else 'forced'})")

    env = SyntheticGridEnv(cfg)
    player = Player(cfg)
    steps = args.steps or int(cfg.get("training.steps", 2000))
    report = player.learn(env, total_steps=steps, checkpoint=args.checkpoint, verbose=not args.no_verbose)

    os.makedirs("runs", exist_ok=True)
    out_path = "runs/phase0_train_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print(f"[phase0_train] report saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
