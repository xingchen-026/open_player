"""Phase 1 evaluation / learning curves.

    python examples/phase1_evaluate.py --checkpoint checkpoints/phase1.pt
    python examples/phase1_evaluate.py --steps 2000   # trains, saving curve checkpoints

Evaluates goal success rate, resource collection, exploration coverage and
1/4/8-step prediction errors at the curve steps (1000/5000/10000/20000 by
default, or whichever steps fit the run).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from open_player.agent.player import Player
from open_player.core.config import load_config, resolve_device, set_seed, setup_logging
from open_player.environments.transfer import make_transfer_envs
from open_player.evaluation.benchmark import evaluate_world_model
from open_player.evaluation.logger import ExperimentLogger


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Player Phase 1 evaluation")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "cuda:0"])
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    setup_logging(cfg.get("logging.level", "INFO"))
    set_seed(int(cfg.seed))
    print(f"[phase1_evaluate] device={resolve_device(cfg)}")

    pair = make_transfer_envs(cfg)
    env_a = pair.world_a
    player = Player(cfg)

    curve_steps = [s for s in cfg.get("evaluation.curve_steps", [1000, 5000, 10000, 20000]) if s <= args.steps]
    log_dir = cfg.get("evaluation.log_dir", "runs/phase1")
    ckpt_dir = "checkpoints/phase1_curve"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoints = {}
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoints["provided"] = args.checkpoint
    else:
        # train with periodic curve checkpoints
        done = 0
        saved = set()
        while done < args.steps:
            n = min(250, args.steps - done)
            player.learn(env_a, total_steps=n, verbose=False)
            done += n
            for s in curve_steps:
                if done >= s and s not in saved:
                    saved.add(s)
                    player.save_checkpoint(os.path.join(ckpt_dir, f"phase1_step{s}.pt"))
        player.save_checkpoint(os.path.join(ckpt_dir, "phase1_final.pt"))
        for s in curve_steps:
            path = os.path.join(ckpt_dir, f"phase1_step{s}.pt")
            if os.path.exists(path):
                checkpoints[f"step{s}"] = path
        checkpoints["final"] = os.path.join(ckpt_dir, "phase1_final.pt")

    logger = ExperimentLogger(log_dir, name="phase1_learning_curve")
    curve = []
    for name, path in checkpoints.items():
        player.load_checkpoint(path)
        ev = player.evaluate(env_a, episodes=args.episodes, max_steps=args.max_steps)
        wm = evaluate_world_model(player.world_model, player.perceive, env_a)
        row = {
            "checkpoint": name,
            "step": player.trainer.step,
            "goal_success_rate": ev.get("goal_success_rate", 0.0),
            "mean_collected": ev.get("mean_collected", 0.0),
            "mean_exploration_coverage": ev.get("mean_exploration_coverage", 0.0),
            "step1_entity": wm.get("step1_entity", 0.0),
            "step4_entity": wm.get("step4_entity", 0.0),
            "step8_entity": wm.get("step8_entity", 0.0),
            "step4_latent": wm.get("step4_latent", 0.0),
            "step8_latent": wm.get("step8_latent", 0.0),
        }
        curve.append(row)
        logger.record(row["step"], **{k: v for k, v in row.items() if k != "step"})
        print(f"[phase1_evaluate] {name:<10} step={row['step']:<6} goal={row['goal_success_rate']:.3f} "
              f"collect={row['mean_collected']:.2f} coverage={row['mean_exploration_coverage']:.3f} "
              f"err1={row['step1_entity']:.4f} err4={row['step4_entity']:.4f} err8={row['step8_entity']:.4f}", flush=True)
    logger.close()

    out_path = os.path.join(log_dir, "learning_curve.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(curve, fh, indent=2)
    print(f"[phase1_evaluate] learning curve -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
