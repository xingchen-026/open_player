"""Phase 1 demo: train World A, freeze, explore World B (unknown map).

    python examples/phase1_demo.py --steps 300
    python examples/phase1_demo.py --checkpoint checkpoints/phase1.pt --render

The model is frozen when entering World B: no World B knowledge is used for
training.  The demo reports prediction errors, exploration coverage, goal
success and resources collected in the unseen world.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from open_player.agent.player import Player
from open_player.core.config import load_config, resolve_device, set_seed, setup_logging
from open_player.environments.transfer import make_transfer_envs, world_structural_summary
from open_player.evaluation.benchmark import evaluate_world_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Player Phase 1 transfer demo")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--run-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "cuda:0"])
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    setup_logging(cfg.get("logging.level", "INFO"))
    set_seed(int(cfg.seed))
    print(f"[phase1_demo] device={resolve_device(cfg)}")

    pair = make_transfer_envs(cfg)
    env_a, env_b = pair.world_a, pair.world_b
    print("[phase1_demo] world A:", world_structural_summary(env_a))
    print("[phase1_demo] world B:", world_structural_summary(env_b))

    player = Player(cfg)
    if args.checkpoint and os.path.exists(args.checkpoint):
        meta = player.load_checkpoint(args.checkpoint)
        print(f"[phase1_demo] loaded checkpoint step={meta['step']}")
    else:
        print(f"[phase1_demo] training on World A for {args.steps} steps ...")
        player.learn(env_a, total_steps=args.steps, verbose=False)

    # freeze: no further weight updates
    player.trainer.eval()
    errs_before = evaluate_world_model(player.world_model, player.perceive, env_b)
    print(f"[phase1_demo] World B prediction errors (zero-shot): "
          f"step1_entity={errs_before.get('step1_entity', 0):.4f} "
          f"step4_entity={errs_before.get('step4_entity', 0):.4f} "
          f"step8_entity={errs_before.get('step8_entity', 0):.4f}", flush=True)

    print("[phase1_demo] entering World B (unknown map, model frozen):", flush=True)
    rep = player.run(env_b, max_steps=args.run_steps, render=args.render, verbose=not args.render)

    w = env_b.world
    free = w.grid_size * w.grid_size - len(w.walls)
    coverage = len(w.visited) / max(free, 1)
    print(f"[phase1_demo] coverage={coverage:.3f} collected={rep.collected_total} "
          f"goal={rep.extra.get('goal_type')} status={rep.extra.get('goal_status')} "
          f"mean_reward={rep.mean_reward:+.3f}")
    print(f"[phase1_demo] events_seen={rep.events} episodes={rep.episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
