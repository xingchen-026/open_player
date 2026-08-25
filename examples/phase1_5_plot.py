"""Phase 1.5 plots (matplotlib): learning curves, error-vs-horizon,
ablation comparisons, transfer/adaptation.

    python examples/phase1_5_plot.py --run-dir results/p1.5-xxxx
    python examples/phase1_5_plot.py   # uses results/latest_run.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _latest(results_dir: str) -> str:
    marker = os.path.join(results_dir, "latest_run.txt")
    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as fh:
            return fh.read().strip()
    dirs = sorted(d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)))
    return dirs[-1] if dirs else ""


def _load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def plot_learning_curve(run_dir: str) -> str:
    path = os.path.join(run_dir, "learning_curve_summary.json")
    if not os.path.exists(path):
        return ""
    s = _load(path)
    steps = [int(k) for k in sorted(s, key=lambda k: int(k))]
    goals = [s[str(k)].get("goal_success_rate_mean", 0) for k in steps]
    gstd = [s[str(k)].get("goal_success_rate_std", 0) for k in steps]
    cov = [s[str(k)].get("coverage_mean", 0) for k in steps]
    cstd = [s[str(k)].get("coverage_std", 0) for k in steps]
    e1 = [s[str(k)].get("step1_entity_mean", np.nan) for k in steps]
    e8 = [s[str(k)].get("step8_entity_mean", np.nan) for k in steps]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.errorbar(steps, goals, yerr=gstd, marker="o", label="goal success")
    ax.errorbar(steps, cov, yerr=cstd, marker="s", label="exploration coverage")
    ax.set_xlabel("training steps"); ax.set_ylabel("rate"); ax.legend(); ax.set_title("Learning curve (mean +/- std over seeds)")
    ax = axes[1]
    ax.plot(steps, e1, marker="o", label="1-step entity error")
    ax.plot(steps, e8, marker="s", label="8-step entity error")
    ax.set_xlabel("training steps"); ax.set_ylabel("MSE"); ax.legend(); ax.set_title("Prediction error vs training")
    out = os.path.join(run_dir, "learning_curve.png")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def plot_world_model_ablation(run_dir: str) -> str:
    path = os.path.join(run_dir, "world_model_ablation_summary.json")
    if not os.path.exists(path):
        return ""
    s = _load(path)
    horizons = [1, 4, 8, 16]
    models = ["persistence", "random", "phase0", "phase1"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, env_name in enumerate(("world_a", "world_b")):
        ax = axes[i]
        for m in models:
            key = f"{m}.{env_name}"
            if key not in s:
                continue
            vals = [s[key].get(f"step{h}_entity_mean", np.nan) for h in horizons]
            ax.plot(horizons, vals, marker="o", label=m)
        ax.set_xlabel("horizon"); ax.set_ylabel("entity MSE"); ax.legend(); ax.set_title(f"error vs horizon ({env_name})")
    out = os.path.join(run_dir, "world_model_ablation.png")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def plot_ablation_bars(run_dir: str) -> str:
    """skill + intrinsic + multistep bars from whichever files exist."""
    figs = []
    # skill ablation
    path = os.path.join(run_dir, "skill_ablation.json")
    if os.path.exists(path):
        rows = _load(path)
        variants = {}
        for r in rows:
            variants.setdefault(r["variant"], []).append(r.get("coverage_mean", 0.0))
        if variants:
            fig, ax = plt.subplots(figsize=(6, 4))
            names = sorted(variants)
            means = [np.mean(variants[n]) for n in names]
            stds = [np.std(variants[n]) for n in names]
            ax.bar(names, means, yerr=stds, capsize=4)
            ax.set_ylabel("exploration coverage"); ax.set_title("NeuralSkill ablation")
            out = os.path.join(run_dir, "skill_ablation.png")
            fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
            figs.append(out)
    # intrinsic ablation
    path = os.path.join(run_dir, "intrinsic_ablation_summary.json")
    if os.path.exists(path):
        s = _load(path)
        variants = sorted(s)
        cov = [s[v].get("coverage_mean", 0) for v in variants]
        death = [s[v].get("death_rate_mean", 0) for v in variants]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(variants, cov); axes[0].set_title("coverage per intrinsic variant"); axes[0].tick_params(axis="x", rotation=30)
        axes[1].bar(variants, death); axes[1].set_title("death rate per intrinsic variant"); axes[1].tick_params(axis="x", rotation=30)
        out = os.path.join(run_dir, "intrinsic_ablation.png")
        fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
        figs.append(out)
    # multistep ablation
    path = os.path.join(run_dir, "multistep_ablation_summary.json")
    if os.path.exists(path):
        s = _load(path)
        variants = sorted(k for k in s if not k.startswith(("h148_vs",)))
        horizons = [1, 4, 8, 16]
        fig, ax = plt.subplots(figsize=(6, 4))
        for v in variants:
            if isinstance(s[v], dict):
                vals = [s[v].get(f"step{h}_latent_mean", np.nan) for h in horizons]
                ax.plot(horizons, vals, marker="o", label=v)
        ax.set_xlabel("horizon"); ax.set_ylabel("latent MSE"); ax.legend(); ax.set_title("multi-step training ablation")
        out = os.path.join(run_dir, "multistep_ablation.png")
        fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
        figs.append(out)
    return ", ".join(figs)


def plot_adaptation(run_dir: str) -> str:
    path = os.path.join(run_dir, "adaptation_summary.json")
    if not os.path.exists(path):
        alt = [f for f in os.listdir(run_dir) if f.startswith("adaptation_") and f.endswith("_summary.json")]
        if not alt:
            return ""
        path = os.path.join(run_dir, sorted(alt)[0])
    s = _load(path)
    agents = ("phase0", "phase1")
    adapts = [0, 100, 500, 1000]
    fig, ax = plt.subplots(figsize=(7, 4))
    for agent in agents:
        xs, ys = [], []
        for a in adapts:
            key = f"{agent}.adapt{a}"
            if key in s:
                xs.append(a); ys.append(s[key].get("collected_mean", 0.0))
        ax.plot(xs, ys, marker="o", label=agent)
    ax.set_xlabel("adaptation samples"); ax.set_ylabel("mean collected"); ax.legend(); ax.set_title("adaptation curve (held-out world)")
    out = os.path.join(run_dir, "adaptation.png")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1.5 plots")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    run_dir = args.run_dir or os.path.join(args.results, _latest(args.results))
    if not os.path.isdir(run_dir):
        print(f"[plot] run dir not found: {run_dir}")
        return 1
    outs = [f for f in (plot_learning_curve(run_dir), plot_world_model_ablation(run_dir), plot_ablation_bars(run_dir), plot_adaptation(run_dir)) if f]
    for f in outs:
        print(f"[plot] {f}")
    if not outs:
        print("[plot] no plottable summaries found in", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
