"""Phase 1 training example: learned vision + multi-step world model.

    python examples/phase1_train.py
    python examples/phase1_train.py --steps 5000 --train-skill
    python examples/phase1_train.py --checkpoint checkpoints/phase1.pt

Trains on World A (open map) with:
* RGB (160x90) -> LearnedVisionEncoder -> WorldState
* 1-step + 4-step + 8-step world model losses (scheduled teacher forcing)
* learned change / boundary prediction
* intrinsic reward feeding goal selection and exploration
* optional NeuralSkill behavior cloning from rule trajectories

Periodically evaluates and writes a CSV/JSONL learning log.
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
    parser = argparse.ArgumentParser(description="Open Player Phase 1 training")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "cuda:0"])
    parser.add_argument("--checkpoint", default="checkpoints/phase1.pt")
    parser.add_argument("--train-skill", action="store_true", help="behavior-clone a NeuralSkill first")
    parser.add_argument("--eval-every", type=int, default=250)
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
    print(f"[phase1_train] device={device} seed={cfg.seed}")

    pair = make_transfer_envs(cfg)
    env_a = pair.world_a
    player = Player(cfg)

    if args.train_skill:
        report = player.train_skill(env_a, steps=int(cfg.get("skill_training.train_steps", 400)))
        print(f"[phase1_train] neural skill trained: acc={report.action_accuracy:.3f} params={report.params}")

    steps = args.steps or int(cfg.get("training.steps", 2000))
    log_dir = cfg.get("evaluation.log_dir", "runs/phase1")
    logger = ExperimentLogger(log_dir, name="phase1_train")
    total_done = 0
    chunk = args.eval_every
    while total_done < steps:
        n = min(chunk, steps - total_done)
        rep = player.learn(env_a, total_steps=n, verbose=not args.no_verbose)
        total_done += n
        ev = player.evaluate(env_a, episodes=int(cfg.get("evaluation.eval_episodes", 4)), max_steps=int(cfg.get("evaluation.eval_max_steps", 100)))
        wm = evaluate_world_model(player.world_model, player.perceive, env_a)
        logger.record(total_done, **{
            "loss": rep.final_loss,
            "intrinsic_mean": rep.extra.get("mean_intrinsic_reward", 0.0),
            "eval": ev,
            "wm": wm,
        })
        print(f"[phase1_train] step {total_done}/{steps} | goal_success_rate={ev.get('goal_success_rate', 0):.3f} "
              f"coverage={ev.get('mean_exploration_coverage', 0):.3f} collected={ev.get('mean_collected', 0):.2f} "
              f"wm_step8_entity={wm.get('step8_entity', float('nan')):.4f}", flush=True)
    logger.close()

    if args.checkpoint:
        path = player.save_checkpoint(args.checkpoint)
        print(f"[phase1_train] checkpoint -> {path}")

    summary_path = os.path.join(log_dir, "phase1_train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({
            "steps": total_done,
            "final_loss": rep.final_loss,
            "model_params": rep.extra.get("model_params"),
            "vision_params": rep.extra.get("vision_params"),
            "neural_skill_params": rep.extra.get("neural_skill_params"),
            "last_eval": ev,
            "last_wm_errors": wm,
        }, fh, indent=2, default=float)
    print(f"[phase1_train] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
