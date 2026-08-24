"""Phase 1 transfer test: train World A, zero-shot World B, short adaptation.

    python examples/phase1_transfer.py --steps 2000 --adaptation-steps 1000

World A (open map, clustered resources, slow enemies) vs World B (narrow
corridors, scattered resources, fast enemies).  The agent never sees World B
during training; zero-shot and post-adaptation performance are compared
against Random / Rule / Phase 0 agent baselines.  Results go to
runs/phase1/transfer_results.json.
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


def _row(results: dict) -> dict:
    """Flatten nested dicts for the transfer curve table."""
    out = {}
    for key in ("baselines_b", "world_a", "training"):
        if key in results and isinstance(results[key], dict):
            for k, v in results[key].items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            out[f"{key}.{k}.{kk}"] = vv
                elif isinstance(v, (int, float)):
                    out[f"{key}.{k}"] = v
    for key in ("world_b",):
        if key in results and isinstance(results[key], dict):
            for k, v in results[key].items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            out[f"{key}.{k}.{kk}"] = vv
    for key in ("prediction_errors_a", "prediction_errors_b_zero_shot", "prediction_errors_b_adapted"):
        if key in results and isinstance(results[key], dict):
            for k, v in results[key].items():
                out[f"{key}.{k}"] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Player Phase 1 transfer test")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--adaptation-steps", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
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
    print(f"[phase1_transfer] device={resolve_device(cfg)}")

    pair = make_transfer_envs(cfg)
    player = Player(cfg)
    if args.checkpoint and os.path.exists(args.checkpoint):
        meta = player.load_checkpoint(args.checkpoint)
        print(f"[phase1_transfer] loaded checkpoint step={meta['step']}")

    adaptation = args.adaptation_steps if args.adaptation_steps is not None else int(cfg.get("evaluation.adaptation_steps", 1000))
    results = player.evaluate_transfer(
        train_env=pair.world_a,
        test_env=pair.world_b,
        steps=args.steps,
        adaptation_steps=adaptation,
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_dir=cfg.get("evaluation.log_dir", "runs/phase1"),
    )

    # readable table
    row = _row(results)
    interesting = [
        "baselines_b.random.mean_collected", "baselines_b.rule.mean_collected",
        "baselines_b.phase0_agent.mean_collected",
        "world_b.zero_shot.mean_collected", "world_b.after_adaptation.mean_collected",
        "baselines_b.random.mean_exploration_coverage", "baselines_b.rule.mean_exploration_coverage",
        "baselines_b.phase0_agent.mean_exploration_coverage",
        "world_b.zero_shot.mean_exploration_coverage", "world_b.after_adaptation.mean_exploration_coverage",
        "prediction_errors_b_zero_shot.step1_entity", "prediction_errors_b_adapted.step1_entity",
    ]
    print("[phase1_transfer] transfer summary:", flush=True)
    for k in interesting:
        if k in row:
            print(f"  {k:<60} {row[k]:.4f}", flush=True)

    curve_path = os.path.join(cfg.get("evaluation.log_dir", "runs/phase1"), "transfer_curve.json")
    with open(curve_path, "w", encoding="utf-8") as fh:
        json.dump(row, fh, indent=2)
    print(f"[phase1_transfer] transfer curve -> {curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
