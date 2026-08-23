"""Phase 0 end-to-end demo: goal -> planner -> skill -> action loop.

Shows the agent completing the simple goal "find a resource and collect it"
inside the synthetic world, with optional ASCII rendering and an optional
pre-trained checkpoint.

Usage:
    python examples/phase0_demo.py                    # fresh model (still works, rule skills)
    python examples/phase0_demo.py --checkpoint checkpoints/phase0.pt
    python examples/phase0_demo.py --render
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from open_player.agent.player import Player
from open_player.core.config import load_config, set_seed, setup_logging
from open_player.environments.synthetic.env import SyntheticGridEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Player Phase 0 end-to-end demo")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.seed = args.seed
    cfg.device = args.device
    setup_logging(cfg.get("logging.level", "INFO"))
    set_seed(int(cfg.seed))

    # A friendlier world for the demo: fewer enemies, more hp, slower enemy.
    cfg.environment.num_enemies = 1
    cfg.environment.num_resources = 3
    cfg.environment.player_hp = 15
    cfg.environment.enemy_move_prob = 0.3
    cfg.environment.enemy_attack_prob = 0.3
    cfg.environment.max_steps = args.steps

    env = SyntheticGridEnv(cfg)
    player = Player(cfg)
    if args.checkpoint and os.path.exists(args.checkpoint):
        meta = player.load_checkpoint(args.checkpoint)
        print(f"[phase0_demo] loaded checkpoint {args.checkpoint} (step={meta['step']})")
    elif args.checkpoint:
        print(f"[phase0_demo] warning: checkpoint not found ({args.checkpoint}); using a fresh model")

    print("[phase0_demo] goal: find a resource and collect it (task goal)")
    report = player.run(env, max_steps=args.steps, render=args.render, verbose=True)

    ok = report.collected_total >= 1
    print("[phase0_demo] result:", "GOAL COMPLETED (collected >= 1 resource)" if ok else "goal not completed in time")
    print(f"[phase0_demo] collected={report.collected_total} steps={report.total_steps} "
          f"mean_reward={report.mean_reward:+.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
