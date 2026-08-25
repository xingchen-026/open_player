"""Phase 1.5 transfer / adaptation / generalization.

    python examples/phase1_5_transfer.py --experiment transfer
    python examples/phase1_5_transfer.py --experiment generalization

Transfer uses the learning-curve checkpoints (run --experiment curve first,
or pass --ckpt-dir).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from open_player.core.config import load_config, resolve_device, set_seed, setup_logging
from open_player.environments.transfer import make_transfer_envs
from open_player.evaluation.experiments import run_generalization, run_transfer_benchmark
from open_player.evaluation.protocol import EvalProtocol
from open_player.evaluation.repro import make_run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Player Phase 1.5 transfer")
    parser.add_argument("--experiment", default="transfer", choices=["transfer", "generalization"])
    parser.add_argument("--config", default="configs/phase1_5.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--base-step", type=int, default=10000)
    parser.add_argument("--adaptation", default="100,500,1000")
    parser.add_argument("--worlds", default=None, help="comma list: world_b,world_c")
    parser.add_argument("--agents", default=None, help="comma list: phase0,phase1")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "cuda:0"])
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device
    setup_logging(cfg.get("logging.level", "INFO"))
    set_seed(int(cfg.seed))
    resolve_device(cfg)
    pair = make_transfer_envs(cfg)
    seeds = [int(s) for s in args.seeds.split(",")]
    adaptation = [int(a) for a in args.adaptation.split(",")]
    protocol = EvalProtocol(seeds=seeds, episodes=args.episodes, max_steps=args.max_steps)
    if args.results_dir and os.path.isdir(args.results_dir):
        run_dir = args.results_dir
    else:
        run_dir = make_run_dir(args.results, config=cfg.to_dict(), note=f"phase1_5:{args.experiment}")
    print(f"[transfer] {args.experiment} -> {run_dir}")

    if args.experiment == "transfer":
        ckpt_dir = args.ckpt_dir or os.path.join("results", _latest(args.results), "checkpoints")
        run_transfer_benchmark(
            cfg, pair, protocol, seeds, adaptation, ckpt_dir, args.base_step, run_dir,
            worlds=[w.strip() for w in args.worlds.split(",")] if args.worlds else None,
            agents=[a.strip() for a in args.agents.split(",")] if args.agents else None,
        )
    else:
        ckpt_dir = args.ckpt_dir or os.path.join("results", _latest(args.results), "checkpoints")
        run_generalization(cfg, protocol, seeds, ckpt_dir, args.base_step, run_dir)
    print("[transfer] done:", run_dir)
    return 0


def _latest(results_dir: str) -> str:
    marker = os.path.join(results_dir, "latest_run.txt")
    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as fh:
            return fh.read().strip()
    dirs = sorted(os.listdir(results_dir))
    return dirs[-1] if dirs else ""


if __name__ == "__main__":
    raise SystemExit(main())
