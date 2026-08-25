"""Phase 1.5 ablations: skill / world model / multi-step / intrinsic /
vision modes / representation.

    python examples/phase1_5_ablation.py --experiment skill
    python examples/phase1_5_ablation.py --experiment worldmodel
    python examples/phase1_5_ablation.py --experiment multistep
    python examples/phase1_5_ablation.py --experiment intrinsic
    python examples/phase1_5_ablation.py --experiment vision
    python examples/phase1_5_ablation.py --experiment representation
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from open_player.core.config import load_config, resolve_device, set_seed, setup_logging
from open_player.environments.transfer import make_transfer_envs
from open_player.evaluation.experiments import (
    run_intrinsic_ablation,
    run_multistep_ablation,
    run_representation_ablation,
    run_skill_ablation,
    run_vision_modes,
    run_world_model_ablation,
)
from open_player.evaluation.protocol import EvalProtocol
from open_player.evaluation.repro import make_run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Player Phase 1.5 ablations")
    parser.add_argument("--experiment", required=True, choices=["skill", "worldmodel", "multistep", "intrinsic", "vision", "representation"])
    parser.add_argument("--config", default="configs/phase1_5.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "cuda:0"])
    parser.add_argument("--variants", default=None, help="comma list (intrinsic: none,novelty,error,novelty_error,full | multistep: h1,h14,h148)")
    parser.add_argument("--models", default=None, help="comma list (worldmodel: persistence,random,phase0,phase1)")
    parser.add_argument("--modes", default=None, help="comma list (vision: structured,side,learned_grid,strict)")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device
    setup_logging(cfg.get("logging.level", "INFO"))
    set_seed(int(cfg.seed))
    device = resolve_device(cfg)
    pair = make_transfer_envs(cfg)
    seeds = [int(s) for s in args.seeds.split(",")]
    protocol = EvalProtocol(seeds=seeds, episodes=args.episodes, max_steps=args.max_steps)
    if args.results_dir and os.path.isdir(args.results_dir):
        run_dir = args.results_dir
    else:
        run_dir = make_run_dir(args.results, config=cfg.to_dict(), note=f"phase1_5:ablation:{args.experiment}")
    print(f"[ablation] {args.experiment} -> {run_dir} (device={device})")

    if args.experiment == "skill":
        run_skill_ablation(cfg, pair.world_a, protocol, seeds, run_dir, bc_steps=min(args.steps, 400))
    elif args.experiment == "worldmodel":
        run_world_model_ablation(cfg, pair, seeds, train_steps=args.steps, run_dir=run_dir, device=str(device),
                                 models=[m.strip() for m in args.models.split(",")] if args.models else None)
    elif args.experiment == "multistep":
        run_multistep_ablation(cfg, pair.world_a, seeds, train_steps=args.steps, run_dir=run_dir, device=str(device),
                               variants=[v.strip() for v in args.variants.split(",")] if args.variants else None)
    elif args.experiment == "intrinsic":
        run_intrinsic_ablation(cfg, pair.world_a, protocol, seeds, steps=args.steps, run_dir=run_dir,
                               variants=[v.strip() for v in args.variants.split(",")] if args.variants else None)
    elif args.experiment == "vision":
        run_vision_modes(cfg, pair, protocol, seeds, steps=args.steps, run_dir=run_dir, ckpt_dir=os.path.join(run_dir, "checkpoints"),
                         modes=[m.strip() for m in args.modes.split(",")] if args.modes else None)
    elif args.experiment == "representation":
        run_representation_ablation(cfg, pair.world_a, seeds, train_steps=args.steps, run_dir=run_dir, device=str(device))
    print("[ablation] done:", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
