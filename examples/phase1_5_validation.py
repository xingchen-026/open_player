"""Phase 1.5 validation: baselines, learning curves, compute report.

    python examples/phase1_5_validation.py --experiment baselines
    python examples/phase1_5_validation.py --experiment curve --max-step 10000 --seeds 0,1,2,3,4
    python examples/phase1_5_validation.py --experiment compute

Results land in results/<experiment_id>/ (CSV/JSONL + config snapshot + git
hash) - fully reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from open_player.core.config import load_config, resolve_device, set_seed, setup_logging
from open_player.environments.transfer import make_transfer_envs
from open_player.evaluation.experiments import run_baseline_comparison, run_learning_curve
from open_player.evaluation.protocol import EvalProtocol
from open_player.evaluation.repro import make_run_dir, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Player Phase 1.5 validation")
    parser.add_argument("--experiment", default="baselines", choices=["baselines", "curve", "compute"])
    parser.add_argument("--config", default="configs/phase1_5.yaml")
    parser.add_argument("--seeds", default=None, help="comma list, e.g. 0,1,2,3,4")
    parser.add_argument("--max-step", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--train-skill", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume the curve from existing checkpoints")
    parser.add_argument("--results-dir", default=None, help="reuse an existing run dir (resume)")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "cuda:0"])
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device
    setup_logging(cfg.get("logging.level", "INFO"))
    set_seed(int(cfg.seed))
    device = resolve_device(cfg)
    pair = make_transfer_envs(cfg)
    seeds = [int(s) for s in (args.seeds or ",".join(str(s) for s in cfg.get("validation.seeds", [0, 1, 2, 3, 4]))).split(",")]
    protocol = EvalProtocol(seeds=seeds, episodes=args.episodes, max_steps=args.max_steps)

    if args.results_dir:
        run_dir = args.results_dir
        if not os.path.isdir(run_dir):
            print(f"[phase1_5] results dir not found: {run_dir}")
            return 1
    else:
        run_dir = make_run_dir(args.results, config=cfg.to_dict(), note=f"phase1_5:{args.experiment}")
    print(f"[phase1_5] run dir: {run_dir} (device={device})")

    if args.experiment == "baselines":
        summary = run_baseline_comparison(cfg, pair, protocol, run_dir)
        for env_name in ("world_a", "world_b"):
            for agent in ("random", "rule", "phase0", "phase1"):
                key = f"{env_name}.{agent}"
                if key in summary:
                    s = summary[key]
                    print(f"  {key:<22} collected={s.get('collected_mean', 0):.2f}+-{s.get('collected_std', 0):.2f} "
                          f"coverage={s.get('coverage_mean', 0):.2f}+-{s.get('coverage_std', 0):.2f}")
    elif args.experiment == "curve":
        curve_steps = [s for s in cfg.get("validation.curve_steps", [1000, 2000, 5000, 10000, 20000]) if s <= args.max_step]
        summary = run_learning_curve(
            cfg, pair.world_a, protocol, seeds, curve_steps,
            run_dir, ckpt_dir=os.path.join(run_dir, "checkpoints"),
            train_skill=args.train_skill,
            resume=args.resume,
        )
        for step in curve_steps:
            if str(step) in summary:
                s = summary[str(step)]
                print(f"  step={step:<6} goal={s.get('goal_success_rate_mean', 0):.3f}+-{s.get('goal_success_rate_std', 0):.3f} "
                      f"collect={s.get('collected_mean', 0):.2f}+-{s.get('collected_std', 0):.2f} "
                      f"cov={s.get('coverage_mean', 0):.2f}+-{s.get('coverage_std', 0):.2f} "
                      f"err1={s.get('step1_entity_mean', 0):.4f} err8={s.get('step8_entity_mean', 0):.4f}")
    elif args.experiment == "compute":
        from open_player.agent.player import Player
        from open_player.skills.neural import NeuralSkill, StateFeaturizer
        report: dict = {"phase0": {}, "phase1": {}}
        # Phase 0
        cfg0 = load_config("configs/phase0.yaml")
        cfg0.device = str(device)
        p0 = Player(cfg0)
        torch.cuda.reset_peak_memory_stats() if str(device).startswith("cuda") else None
        t0 = time.time()
        p0.learn(pair.world_a, total_steps=100, verbose=False)
        report["phase0"] = {
            "params": p0.world_model.num_parameters(),
            "steps_per_sec": 100 / max(time.time() - t0, 1e-6),
            "gpu_mb_peak": torch.cuda.max_memory_allocated() / 1e6 if str(device).startswith("cuda") else 0.0,
        }
        # Phase 1 (side)
        p1 = Player(cfg)
        torch.cuda.reset_peak_memory_stats() if str(device).startswith("cuda") else None
        t0 = time.time()
        p1.learn(pair.world_a, total_steps=100, verbose=False)
        feat = StateFeaturizer(p1.schema, device=str(device))
        skill = NeuralSkill("n", list(pair.world_a.action_space.names), featurizer=feat)
        report["phase1"] = {
            "params_world_model": p1.world_model.num_parameters(),
            "params_vision": p1.vision.num_parameters(),
            "params_change": p1.world_model.change_predictor.num_parameters(),
            "params_skill": skill.num_parameters(),
            "params_total": p1.world_model.num_parameters() + p1.vision.num_parameters() + p1.world_model.change_predictor.num_parameters() + skill.num_parameters(),
            "steps_per_sec": 100 / max(time.time() - t0, 1e-6),
            "gpu_mb_peak": torch.cuda.max_memory_allocated() / 1e6 if str(device).startswith("cuda") else 0.0,
        }
        ckpt = os.path.join(run_dir, "phase1.pt")
        p1.save_checkpoint(ckpt)
        report["phase1"]["checkpoint_bytes"] = os.path.getsize(ckpt)
        save_json(run_dir, "compute_report.json", report)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
