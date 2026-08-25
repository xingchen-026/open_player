"""Phase 1.5 experiment functions (the examples are thin CLI wrappers).

Every experiment:
* uses the SAME EvalProtocol (envs, budgets, seed set);
* writes CSV/JSONL rows + JSON aggregates into a reproducible run dir;
* reports mean/std/median, never single-run numbers;
* uses held-out trajectories / worlds for model evaluation.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from open_player.agent.player import Player
from open_player.core.config import Config
from open_player.core.schema import SchemaSet
from open_player.core.types import Observation, WorldState
from open_player.environments.synthetic.env import SyntheticGridEnv
from open_player.evaluation.baselines import RandomBaseline, RuleBaseline
from open_player.evaluation.protocol import EvalProtocol, aggregate_rows
from open_player.evaluation.repro import save_csv, save_json
from open_player.evaluation.stats import is_improvement
from open_player.evaluation.world_model_baselines import PersistenceWorldModel, RandomDynamicsWorldModel
from open_player.tracking.tracker import BeliefTracker
from open_player.world.model import WorldModel


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def collect_sequence(
    env: SyntheticGridEnv,
    perceive: Callable[[Observation, int], WorldState],
    length: int = 17,
    seed: int = 0,
    action_fn: Optional[Callable[[WorldState], int]] = None,
) -> Tuple[List[WorldState], List[int]]:
    """A held-out (state, action) trajectory of the given length."""
    obs = env.reset(seed=seed)
    states = [perceive(obs, 0)]
    actions: List[int] = []
    for i in range(length - 1):
        a = int((i * 3 + 1) % env.action_space.n) if action_fn is None else int(action_fn(states[-1]))
        obs2, _, _, _ = env.step(a)
        states.append(perceive(obs2, i + 1))
        actions.append(a)
    return states, actions


def wm_error_rows(model: Any, states: List[WorldState], actions: List[int], horizons=(1, 4, 8, 16), prefix: str = "") -> Dict[str, float]:
    errs = model.prediction_errors(states[0], actions, states[1:], horizons=horizons)
    return {f"{prefix}{k}": float(v) for k, v in errs.items()}


def player_summary_row(player: Player, env: SyntheticGridEnv, protocol: EvalProtocol, seed: int, prefix: str = "") -> Dict[str, Any]:
    s = player.evaluate(env, episodes=protocol.episodes, max_steps=protocol.max_steps, seed=seed)
    return {f"{prefix}{k}": v for k, v in s.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _append_json_rows(run_dir: str, name: str, rows: List[Dict[str, Any]]) -> None:
    """Accumulate rows across chunked calls; JSON is the source of truth."""
    path = os.path.join(run_dir, name)
    existing: List[Dict[str, Any]] = []
    if os.path.exists(path):
        import json
        with open(path, encoding="utf-8") as fh:
            existing = json.load(fh)
    existing.extend(rows)
    with open(path, "w", encoding="utf-8") as fh:
        import json
        json.dump(existing, fh, indent=2, default=float)


def _append_csv_rows(run_dir: str, name: str, rows: List[Dict[str, Any]]) -> None:
    """Rewrite the CSV from the merged JSON rows (append-safe)."""
    path = os.path.join(run_dir, name)
    import csv
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _merged_rows(run_dir: str, name: str) -> List[Dict[str, Any]]:
    import json
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def new_player(cfg: Config, mode: Optional[str] = None) -> Player:
    if mode is not None:
        cfg = Config(cfg.to_dict())
        cfg.vision.mode = mode
        if mode == "structured":
            cfg.vision.enabled = False
    return Player(cfg)


def _ckpt_path(ckpt_dir: str, seed: int, step: int) -> str:
    return os.path.join(ckpt_dir, f"seed{seed}_step{step}.pt")


def _evaluate_curve_checkpoint(player: Player, env: SyntheticGridEnv, protocol: EvalProtocol, seed: int, step: int) -> Dict[str, Any]:
    s = player.evaluate(env, episodes=protocol.episodes, max_steps=protocol.max_steps, seed=seed)
    wm: Dict[str, float] = {}
    try:
        from open_player.evaluation.benchmark import evaluate_world_model
        wm = evaluate_world_model(player.world_model, player.perceive, env, steps=16, seed=seed)
    except Exception:  # pragma: no cover
        pass
    row: Dict[str, Any] = {
        "seed": seed,
        "step": step,
        "goal_success_rate": s.get("goal_success_rate", 0.0),
        "collected": s.get("mean_collected", 0.0),
        "coverage": s.get("mean_exploration_coverage", 0.0),
        "death_rate": s.get("rate_death", 0.0),
        "intrinsic_mean": player.trainer.latest.get("entity", 0.0),
    }
    for k in ("step1_entity", "step4_entity", "step8_entity", "step16_entity", "step4_latent", "step8_latent", "step16_latent"):
        if k in wm:
            row[k] = wm[k]
    return row


def evaluate_curve_checkpoints(
    cfg: Config,
    env: SyntheticGridEnv,
    protocol: EvalProtocol,
    seeds: List[int],
    curve_steps: List[int],
    ckpt_dir: str,
    run_dir: str,
) -> Dict[str, Any]:
    """(Re-)evaluate saved curve checkpoints (exact per-step models)."""
    for seed in seeds:
        for step in curve_steps:
            path = _ckpt_path(ckpt_dir, seed, step)
            if not os.path.exists(path):
                continue
            p = new_player(cfg, "side")
            p.load_checkpoint(path)
            row = _evaluate_curve_checkpoint(p, env, protocol, seed, step)
            _append_json_rows(run_dir, "learning_curve.json", [row])
            print(f"[curve-eval] seed={seed} step={step} goal={row['goal_success_rate']:.2f} "
                  f"collect={row['collected']:.2f} cov={row['coverage']:.2f} "
                  f"err1={row.get('step1_entity', float('nan')):.4f} err8={row.get('step8_entity', float('nan')):.4f}", flush=True)
    all_rows = _merged_rows(run_dir, "learning_curve.json")
    _append_csv_rows(run_dir, "learning_curve.csv", all_rows)
    summary = {}
    for step in curve_steps:
        sub = [r for r in all_rows if r["step"] == step]
        if sub:
            summary[str(step)] = aggregate_rows(sub)
    save_json(run_dir, "learning_curve_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# E1: fixed baseline comparison (Random / Rule / Phase 0 / Phase 1)
# --------------------------------------------------------------------------- #
def run_baseline_comparison(cfg: Config, pair: Any, protocol: EvalProtocol, run_dir: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    schema = SchemaSet.from_config(cfg)
    for env_name, env in (("world_a", pair.world_a), ("world_b", pair.world_b)):
        tracker = BeliefTracker(schema)

        def perceive(obs, t):
            return tracker.track(None, obs, t)

        for seed in protocol.seeds:
            for kind in ("random", "rule"):
                if kind == "random":
                    policy = RandomBaseline(env.action_space, seed=seed)
                else:
                    policy = RuleBaseline(env.action_space, schema, seed=seed)
                entries = []
                for ep in range(protocol.episodes):
                    obs = env.reset(seed=seed + 100 * ep)
                    state = perceive(obs, 0)
                    total_reward = 0.0
                    info: Dict[str, Any] = {}
                    for t in range(protocol.max_steps):
                        action = policy.act(state)
                        obs2, reward, done, info = env.step(action)
                        total_reward += float(reward)
                        state = perceive(obs2, t + 1)
                        if done:
                            break
                    w = env.world
                    free = w.grid_size * w.grid_size - len(w.walls)
                    entries.append({
                        "collected": int(info.get("collected", 0)),
                        "coverage": len(w.visited) / max(free, 1),
                        "steps": t + 1,
                        "death": int(bool(info.get("death", False))),
                    })
                agg = aggregate_rows(entries)
                row = {
                    "env": env_name, "agent": kind, "seed": seed,
                    "collected": agg.get("collected_mean", 0.0),
                    "coverage": agg.get("coverage_mean", 0.0),
                    "death": agg.get("death_mean", 0.0),
                    "steps": agg.get("steps_mean", 0.0),
                }
                rows.append(row)
        # phase0 + phase1 agents via the unified Player
        for mode, label in (("structured", "phase0"), ("side", "phase1")):
            for seed in protocol.seeds:
                p = new_player(cfg, mode)
                s = p.evaluate(env, episodes=protocol.episodes, max_steps=protocol.max_steps, seed=seed)
                row = {
                    "env": env_name, "agent": label, "seed": seed,
                    "collected": s.get("mean_collected", 0.0),
                    "coverage": s.get("mean_exploration_coverage", 0.0),
                    "death": s.get("rate_death", 0.0),
                    "goal_success_rate": s.get("goal_success_rate", 0.0),
                    "steps": s.get("mean_steps", 0.0),
                }
                rows.append(row)
    save_csv(run_dir, "baselines.csv", rows)
    save_json(run_dir, "baselines.json", rows)
    summary = {}
    for env_name in ("world_a", "world_b"):
        for agent in ("random", "rule", "phase0", "phase1"):
            sub = [r for r in rows if r["env"] == env_name and r["agent"] == agent]
            if sub:
                summary[f"{env_name}.{agent}"] = aggregate_rows(sub)
    save_json(run_dir, "baselines_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# E2: multi-seed learning curves
# --------------------------------------------------------------------------- #
def run_learning_curve(
    cfg: Config,
    env: SyntheticGridEnv,
    protocol: EvalProtocol,
    seeds: List[int],
    curve_steps: List[int],
    run_dir: str,
    ckpt_dir: str,
    train_skill: bool = False,
    chunk: int = 500,
    resume: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    os.makedirs(ckpt_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        sub = Config(cfg.to_dict())
        sub.seed = seed
        player = new_player(sub)
        if train_skill and sub.get("vision.mode", "side") == "side":
            player.train_skill(env, steps=int(sub.get("skill_training.train_steps", 400)), verbose=False)
        done = 0
        saved: Dict[int, str] = {}
        # resume: continue from the newest existing checkpoint for this seed
        if resume:
            existing = sorted([s for s in curve_steps if os.path.exists(_ckpt_path(ckpt_dir, seed, s))])
            if existing:
                last = existing[-1]
                player.load_checkpoint(_ckpt_path(ckpt_dir, seed, last))
                done = player.trainer.step
                saved[last] = _ckpt_path(ckpt_dir, seed, last)
                print(f"[curve] seed={seed} resumed from step {done}", flush=True)
        next_target = 0
        while next_target < len(curve_steps) and done >= curve_steps[next_target]:
            next_target += 1
        while done < max(curve_steps):
            n = min(chunk, max(curve_steps) - done)
            player.learn(env, total_steps=n, verbose=False)
            done += n
            while next_target < len(curve_steps) and done >= curve_steps[next_target]:
                step = curve_steps[next_target]
                path = _ckpt_path(ckpt_dir, seed, step)
                player.save_checkpoint(path)
                saved[step] = path
                next_target += 1
                # evaluate THIS checkpoint now (not the final model)
                row = _evaluate_curve_checkpoint(player, env, protocol, seed, step)
                rows.append(row)
                if verbose:
                    print(f"[curve] seed={seed} step={step} goal={row['goal_success_rate']:.2f} "
                          f"collect={row['collected']:.2f} cov={row['coverage']:.2f} "
                          f"err1={row.get('step1_entity', float('nan')):.4f} err8={row.get('step8_entity', float('nan')):.4f}", flush=True)
                # incremental persistence: long runs survive interruption
                _append_json_rows(run_dir, "learning_curve.json", [row])
    all_rows = _merged_rows(run_dir, "learning_curve.json")
    _append_csv_rows(run_dir, "learning_curve.csv", all_rows)
    summary = {}
    for step in curve_steps:
        sub = [r for r in all_rows if r["step"] == step]
        if sub:
            summary[str(step)] = aggregate_rows(sub)
    save_json(run_dir, "learning_curve_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# E4: adaptation curves (zero-shot -> 100/500/1000 on held-out worlds)
# --------------------------------------------------------------------------- #
def run_adaptation(
    cfg: Config,
    test_env: SyntheticGridEnv,
    protocol: EvalProtocol,
    seeds: List[int],
    adaptation_steps: List[int],
    ckpt_dir: str,
    base_step: int,
    run_dir: str,
    agents: Optional[List[str]] = None,
    world_name: str = "world",
) -> Dict[str, Any]:
    agent_set = set(agents) if agents else {"phase0", "phase1"}
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        if "phase1" in agent_set:
            path = _ckpt_path(ckpt_dir, seed, base_step)
            if os.path.exists(path):
                p1 = new_player(cfg, "side")
                p1.load_checkpoint(path)
                zs = p1.evaluate(test_env, episodes=protocol.episodes, max_steps=protocol.max_steps, seed=seed)
                rows.append({"agent": "phase1", "seed": seed, "adapt": 0, "collected": zs.get("mean_collected", 0.0), "coverage": zs.get("mean_exploration_coverage", 0.0), "goal_success_rate": zs.get("goal_success_rate", 0.0)})
                cum = 0
                for a in adaptation_steps:
                    p1.learn(test_env, total_steps=a - cum, verbose=False)
                    cum = a
                    s = p1.evaluate(test_env, episodes=protocol.episodes, max_steps=protocol.max_steps, seed=seed)
                    rows.append({"agent": "phase1", "seed": seed, "adapt": a, "collected": s.get("mean_collected", 0.0), "coverage": s.get("mean_exploration_coverage", 0.0), "goal_success_rate": s.get("goal_success_rate", 0.0)})
        if "phase0" in agent_set:
            p0 = new_player(cfg, "structured")
            zs0 = p0.evaluate(test_env, episodes=protocol.episodes, max_steps=protocol.max_steps, seed=seed)
            rows.append({"agent": "phase0", "seed": seed, "adapt": 0, "collected": zs0.get("mean_collected", 0.0), "coverage": zs0.get("mean_exploration_coverage", 0.0), "goal_success_rate": zs0.get("goal_success_rate", 0.0)})
            cum = 0
            for a in adaptation_steps:
                p0.learn(test_env, total_steps=a - cum, verbose=False)
                cum = a
                s = p0.evaluate(test_env, episodes=protocol.episodes, max_steps=protocol.max_steps, seed=seed)
                rows.append({"agent": "phase0", "seed": seed, "adapt": a, "collected": s.get("mean_collected", 0.0), "coverage": s.get("mean_exploration_coverage", 0.0), "goal_success_rate": s.get("goal_success_rate", 0.0)})
    fname = f"adaptation_{world_name}"
    _append_json_rows(run_dir, f"{fname}.json", rows)
    all_rows = _merged_rows(run_dir, f"{fname}.json")
    _append_csv_rows(run_dir, f"{fname}.csv", all_rows)
    summary = {}
    for agent in ("phase0", "phase1"):
        for a in [0] + list(adaptation_steps):
            sub = [r for r in all_rows if r["agent"] == agent and r["adapt"] == a]
            if sub:
                summary[f"{agent}.adapt{a}"] = aggregate_rows(sub)
    save_json(run_dir, f"{fname}_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# E5: four vision modes (structured / side / learned_grid / strict)
# --------------------------------------------------------------------------- #
def run_vision_modes(
    cfg: Config,
    pair: Any,
    protocol: EvalProtocol,
    seeds: List[int],
    steps: int,
    run_dir: str,
    ckpt_dir: str,
    modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    os.makedirs(ckpt_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    mode_list = list(modes) if modes else ["structured", "side", "learned_grid", "strict"]
    for mode in mode_list:
        for seed in seeds:
            sub = Config(cfg.to_dict())
            sub.seed = seed
            if mode == "strict":
                # stability overrides for the strict pipeline (documented in
                # the validation report): 1-step training only + a lower
                # learning rate while the learned localisation is still noisy
                sub.multi_step.horizons = []
                sub.training.learning_rate = 0.0005
            player = new_player(sub, mode)
            player.learn(pair.world_a, total_steps=steps, verbose=False)
            for env_name, env in (("world_a", pair.world_a), ("world_b", pair.world_b)):
                s = player.evaluate(env, episodes=protocol.episodes, max_steps=protocol.max_steps, seed=seed)
                wm = {}
                try:
                    from open_player.evaluation.benchmark import evaluate_world_model
                    wm = evaluate_world_model(player.world_model, player.perceive, env, steps=16, seed=seed)
                except Exception:  # pragma: no cover
                    pass
                row = {
                    "mode": mode, "seed": seed, "env": env_name,
                    "goal_success_rate": s.get("goal_success_rate", 0.0),
                    "collected": s.get("mean_collected", 0.0),
                    "coverage": s.get("mean_exploration_coverage", 0.0),
                    "death_rate": s.get("rate_death", 0.0),
                }
                for k in ("step1_entity", "step4_entity", "step8_entity", "step16_entity"):
                    if k in wm:
                        row[k] = wm[k]
                rows.append(row)
            player.save_checkpoint(_ckpt_path(ckpt_dir, seed, 0).replace("_step0", f"_mode_{mode}"))
            print(f"[vision_modes] mode={mode} seed={seed} done", flush=True)
    _append_json_rows(run_dir, "vision_modes.json", rows)
    all_rows = _merged_rows(run_dir, "vision_modes.json")
    _append_csv_rows(run_dir, "vision_modes.csv", all_rows)
    summary = {}
    for mode in mode_list:
        for env_name in ("world_a", "world_b"):
            sub = [r for r in all_rows if r["mode"] == mode and r["env"] == env_name]
            if sub:
                summary[f"{mode}.{env_name}"] = aggregate_rows(sub)
    save_json(run_dir, "vision_modes_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# E6: NeuralSkill ablation (rule / BC / random-init / shuffled-labels)
# --------------------------------------------------------------------------- #
def run_skill_ablation(
    cfg: Config,
    env: SyntheticGridEnv,
    protocol: EvalProtocol,
    seeds: List[int],
    run_dir: str,
    bc_steps: int = 400,
) -> Dict[str, Any]:
    from open_player.evaluation.baselines import RuleBaseline
    from open_player.skills.neural import NeuralSkill, StateFeaturizer
    from open_player.training.skill_trainer import SkillTrainer
    schema = SchemaSet.from_config(cfg)
    rows: List[Dict[str, Any]] = []

    def run_skill(skill, seed):
        tracker = BeliefTracker(schema)

        def perceive(obs, t):
            return tracker.track(None, obs, t)

        entries = []
        for ep in range(protocol.episodes):
            obs = env.reset(seed=seed + 100 * ep)
            state = perceive(obs, 0)
            collected = 0
            for t in range(protocol.max_steps):
                action = skill.act(state)
                obs2, _r, done, info = env.step(action)
                collected = max(collected, int(info.get("collected", 0)))
                state = perceive(obs2, t + 1)
                if done:
                    break
                if skill.should_terminate(state):
                    skill.reset()
            w = env.world
            free = w.grid_size * w.grid_size - len(w.walls)
            entries.append({"collected": collected, "coverage": len(w.visited) / max(free, 1)})
        return aggregate_rows(entries)

    from open_player.core.config import resolve_device
    dev = resolve_device(cfg)
    for seed in seeds:
        sub = Config(cfg.to_dict())
        sub.seed = seed
        featurizer = StateFeaturizer(schema, device=str(dev))
        trainer = SkillTrainer(sub, featurizer, device=str(dev))
        tracker = BeliefTracker(schema)
        policy = RuleBaseline(env.action_space, schema, seed=seed)

        def perceive(obs, t):
            return tracker.track(None, obs, t)

        xs, actions, terms, _ = trainer.collect(env, perceive, policy.act, steps=bc_steps, seed=seed)
        names = list(env.action_space.names)
        # BC
        bc = NeuralSkill("bc", names, featurizer=featurizer).to(dev)
        trainer.train(bc, xs, actions, terms)
        agg = run_skill(bc, seed)
        rows.append({"seed": seed, "variant": "bc", **agg, "train_acc": trainer.last_acc})
        # shuffled labels
        rng = np.random.default_rng(seed)
        shuffled = list(actions)
        rng.shuffle(shuffled)
        sh = NeuralSkill("shuffled", names, featurizer=featurizer).to(dev)
        trainer.train(sh, xs, shuffled, terms)
        agg = run_skill(sh, seed)
        rows.append({"seed": seed, "variant": "shuffled", **agg, "train_acc": trainer.last_acc})
        # random init
        rand = NeuralSkill("random", names, featurizer=featurizer).to(dev)
        agg = run_skill(rand, seed)
        rows.append({"seed": seed, "variant": "random", **agg, "train_acc": 0.0})
        # rule teacher
        entries = []
        for ep in range(protocol.episodes):
            obs = env.reset(seed=seed + 100 * ep)
            state = perceive(obs, 0)
            collected = 0
            for t in range(protocol.max_steps):
                action = policy.act(state)
                obs2, _r, done, info = env.step(action)
                collected = max(collected, int(info.get("collected", 0)))
                state = perceive(obs2, t + 1)
                if done:
                    break
            w = env.world
            free = w.grid_size * w.grid_size - len(w.walls)
            entries.append({"collected": collected, "coverage": len(w.visited) / max(free, 1)})
        rows.append({"seed": seed, "variant": "rule", **aggregate_rows(entries), "train_acc": 1.0})
    save_csv(run_dir, "skill_ablation.csv", rows)
    save_json(run_dir, "skill_ablation.json", rows)
    stats = {}
    bc_cov = [r["coverage_mean"] for r in rows if r["variant"] == "bc"]
    rand_cov = [r["coverage_mean"] for r in rows if r["variant"] == "random"]
    shuf_cov = [r["coverage_mean"] for r in rows if r["variant"] == "shuffled"]
    if len(bc_cov) >= 2:
        stats["bc_vs_random"] = is_improvement(bc_cov, rand_cov, higher_is_better=True)
        stats["bc_vs_shuffled"] = is_improvement(bc_cov, shuf_cov, higher_is_better=True)
    save_json(run_dir, "skill_ablation_stats.json", stats)
    return stats


# --------------------------------------------------------------------------- #
# E7: world model ablation (persistence / random / phase0 / phase1)
# --------------------------------------------------------------------------- #
def run_world_model_ablation(
    cfg: Config,
    pair: Any,
    seeds: List[int],
    train_steps: int,
    run_dir: str,
    device: Any = "cpu",
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from open_player.training.trainer import WorldModelTrainer
    schema = SchemaSet.from_config(cfg)
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        tracker = BeliefTracker(schema, device=device)

        def perceive(obs, t):
            return tracker.track(None, obs, t)

        # training data from World A (rule policy)
        rule = RuleBaseline(pair.world_a.action_space, schema, seed=seed)
        train_states: List[WorldState] = []
        train_actions: List[int] = []
        obs = pair.world_a.reset(seed=seed)
        state = perceive(obs, 0)
        while len(train_states) < train_steps + 16:
            a = rule.act(state).index
            obs2, _r, done, _i = pair.world_a.step(a)
            state2 = perceive(obs2, len(train_states) + 1)
            train_states.append(state)
            train_actions.append(a)
            state = state2
            if done:
                obs = pair.world_a.reset()
                state = perceive(obs, len(train_states))

        model_filter = set(models) if models else {"persistence", "random", "phase0", "phase1"}
        model_dict = {
            "persistence": PersistenceWorldModel(schema),
            "random": RandomDynamicsWorldModel(schema, cfg, num_actions=6, device=device),
        }
        if "persistence" not in model_filter:
            model_dict.pop("persistence", None)
        if "random" not in model_filter:
            model_dict.pop("random", None)
        for label in ("phase0", "phase1"):
            if label not in model_filter:
                continue
            mcfg = Config(cfg.to_dict())
            if label == "phase0":
                mcfg.multi_step.horizons = []
                mcfg.event_pred.enabled = False
                mcfg.vision.enabled = False
            model = WorldModel(schema, mcfg, num_actions=6).to(device)
            trainer = WorldModelTrainer(model, mcfg, schema, device=device)
            for i in range(train_steps):
                trainer.online_step(train_states[i], train_actions[i], train_states[i + 1], 0.0, False, 0.0)
            model_dict[label] = model

        for model_name, model in model_dict.items():
            for env_name, env in (("world_a", pair.world_a), ("world_b", pair.world_b)):
                for hs in range(3):
                    st, ac = collect_sequence(env, perceive, length=17, seed=1000 + 10 * seed + hs)
                    errs = wm_error_rows(model, st, ac, horizons=(1, 4, 8, 16))
                    row = {"seed": seed, "model": model_name, "env": env_name, "traj": hs}
                    row.update(errs)
                    rows.append(row)
        print(f"[wm_ablation] seed={seed} done", flush=True)
    _append_json_rows(run_dir, "world_model_ablation.json", rows)
    all_rows = _merged_rows(run_dir, "world_model_ablation.json")
    _append_csv_rows(run_dir, "world_model_ablation.csv", all_rows)
    summary = {}
    for model_name in ("persistence", "random", "phase0", "phase1"):
        for env_name in ("world_a", "world_b"):
            sub = [r for r in all_rows if r["model"] == model_name and r["env"] == env_name]
            if sub:
                summary[f"{model_name}.{env_name}"] = aggregate_rows(sub)
    save_json(run_dir, "world_model_ablation_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# E8: multi-step training ablation (held-out long-horizon errors)
# --------------------------------------------------------------------------- #
def run_multistep_ablation(
    cfg: Config,
    env: SyntheticGridEnv,
    seeds: List[int],
    train_steps: int,
    run_dir: str,
    device: Any = "cpu",
    variants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from open_player.training.trainer import WorldModelTrainer
    schema = SchemaSet.from_config(cfg)
    all_variants = {"h1": [], "h14": [4], "h148": [4, 8]}
    variant_filter = set(variants) if variants else set(all_variants)
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        tracker = BeliefTracker(schema, device=device)

        def perceive(obs, t):
            return tracker.track(None, obs, t)

        rule = RuleBaseline(env.action_space, schema, seed=seed)
        train_states: List[WorldState] = []
        train_actions: List[int] = []
        obs = env.reset(seed=seed)
        state = perceive(obs, 0)
        while len(train_states) < train_steps + 16:
            a = rule.act(state).index
            obs2, _r, done, _i = env.step(a)
            state2 = perceive(obs2, len(train_states) + 1)
            train_states.append(state)
            train_actions.append(a)
            state = state2
            if done:
                obs = env.reset()
                state = perceive(obs, len(train_states))
        # held-out trajectories
        held = []
        for hs in range(3):
            held.append(collect_sequence(env, perceive, length=17, seed=2000 + 10 * seed + hs))
        for vname, horizons in all_variants.items():
            if vname not in variant_filter:
                continue
            mcfg = Config(cfg.to_dict())
            mcfg.multi_step.horizons = list(horizons)
            mcfg.event_pred.enabled = False
            mcfg.vision.enabled = False
            model = WorldModel(schema, mcfg, num_actions=6).to(device)
            trainer = WorldModelTrainer(model, mcfg, schema, device=device)
            for i in range(train_steps):
                trainer.online_step(train_states[i], train_actions[i], train_states[i + 1], 0.0, False, 0.0)
            for hi, (st, ac) in enumerate(held):
                errs = wm_error_rows(model, st, ac, horizons=(1, 4, 8, 16))
                row = {"seed": seed, "variant": vname, "traj": hi}
                row.update(errs)
                rows.append(row)
        print(f"[ms_ablation] seed={seed} done", flush=True)
    _append_json_rows(run_dir, "multistep_ablation.json", rows)
    all_rows = _merged_rows(run_dir, "multistep_ablation.json")
    _append_csv_rows(run_dir, "multistep_ablation.csv", all_rows)
    stats = {}
    for v in ("h1", "h14", "h148"):
        sub = [r for r in all_rows if r["variant"] == v]
        if sub:
            stats[v] = aggregate_rows(sub)
    save_json(run_dir, "multistep_ablation_summary.json", stats)
    h1 = [r["step16_latent"] for r in all_rows if r["variant"] == "h1"]
    h148 = [r["step16_latent"] for r in all_rows if r["variant"] == "h148"]
    if len(h1) >= 2:
        stats["h148_vs_h1_step16_latent"] = is_improvement(h148, h1, higher_is_better=False)
    save_json(run_dir, "multistep_ablation_stats.json", stats)
    return stats


# --------------------------------------------------------------------------- #
# E9: intrinsic reward ablation (+ curiosity safety metrics)
# --------------------------------------------------------------------------- #
def run_intrinsic_ablation(
    cfg: Config,
    env: SyntheticGridEnv,
    protocol: EvalProtocol,
    seeds: List[int],
    steps: int,
    run_dir: str,
    variants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    all_variants = {
        "none": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0},
        "novelty": {"alpha": 0.0, "beta": 0.3, "gamma": 0.0},
        "error": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        "novelty_error": {"alpha": 1.0, "beta": 0.3, "gamma": 0.0},
        "full": {"alpha": 1.0, "beta": 0.3, "gamma": 0.2},
    }
    variant_filter = set(variants) if variants else set(all_variants)
    rows: List[Dict[str, Any]] = []
    for vname, overrides in all_variants.items():
        if vname not in variant_filter:
            continue
        for seed in seeds:
            sub = Config(cfg.to_dict())
            sub.seed = seed
            for k, v in overrides.items():
                sub.intrinsic[k] = v
            player = new_player(sub, "side")
            trace: Dict[str, Any] = {"damage": 0, "collision": 0, "deaths": 0, "repeat": 0, "n": 0, "intrinsic_sum": 0.0}
            recent: List[tuple] = []

            def cb(step, state, env_info, events, action, info):
                trace["n"] += 1
                types = [e.type for e in events]
                trace["damage"] += int("damage" in types)
                trace["collision"] += int("collision" in types)
                trace["deaths"] += int("death" in types)
                trace["intrinsic_sum"] += float(env_info.get("intrinsic_reward", 0.0))
                key = (tuple(sorted(state.entity_ids[:3])), action.index)
                if key in recent:
                    trace["repeat"] += 1
                recent.append(key)
                if len(recent) > 4:
                    recent.pop(0)

            player.learn(env, total_steps=steps, verbose=False, step_callback=cb)
            s = player.evaluate(env, episodes=protocol.episodes, max_steps=protocol.max_steps, seed=seed)
            n = max(trace["n"], 1)
            w = env.world
            free = w.grid_size * w.grid_size - len(w.walls)
            rows.append({
                "variant": vname,
                "seed": seed,
                "coverage": s.get("mean_exploration_coverage", 0.0),
                "collected": s.get("mean_collected", 0.0),
                "goal_success_rate": s.get("goal_success_rate", 0.0),
                "death_rate": trace["deaths"] / n,
                "collision_rate": trace["collision"] / n,
                "damage_rate": trace["damage"] / n,
                "repeat_trajectory_ratio": trace["repeat"] / n,
                "unique_visited": len(player.visit_counter.counts) / max(free, 1),
                "intrinsic_mean": trace["intrinsic_sum"] / n,
            })
            print(f"[intrinsic] variant={vname} seed={seed} done", flush=True)
    _append_json_rows(run_dir, "intrinsic_ablation.json", rows)
    all_rows = _merged_rows(run_dir, "intrinsic_ablation.json")
    _append_csv_rows(run_dir, "intrinsic_ablation.csv", all_rows)
    summary = {}
    for v in all_variants:
        sub = [r for r in all_rows if r["variant"] == v]
        if sub:
            summary[v] = aggregate_rows(sub)
    save_json(run_dir, "intrinsic_ablation_summary.json", summary)
    stats = {}
    base = [r["coverage"] for r in all_rows if r["variant"] == "none"]
    full = [r["coverage"] for r in all_rows if r["variant"] == "full"]
    if len(base) >= 2:
        stats["full_vs_none_coverage"] = is_improvement(full, base, higher_is_better=True)
        stats["full_vs_none_death"] = is_improvement([r["death_rate"] for r in all_rows if r["variant"] == "full"], [r["death_rate"] for r in all_rows if r["variant"] == "none"], higher_is_better=False)
    save_json(run_dir, "intrinsic_ablation_stats.json", stats)
    return stats


# --------------------------------------------------------------------------- #
# E10: representation ablation (pooled vs per-entity probe accuracy)
# --------------------------------------------------------------------------- #
def run_representation_ablation(
    cfg: Config,
    env: SyntheticGridEnv,
    seeds: List[int],
    train_steps: int,
    run_dir: str,
    device: Any = "cpu",
) -> Dict[str, Any]:
    schema = SchemaSet.from_config(cfg)
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        tracker = BeliefTracker(schema, device=device)

        def perceive(obs, t):
            return tracker.track(None, obs, t)

        rule = RuleBaseline(env.action_space, schema, seed=seed)
        obs = env.reset(seed=seed)
        state = perceive(obs, 0)
        states: List[WorldState] = []
        for i in range(train_steps):
            a = rule.act(state).index
            obs2, _r, done, _i = env.step(a)
            states.append(state)
            state = perceive(obs2, i + 1)
            if done:
                obs = env.reset()
                state = perceive(obs, len(states))
        # train a world model so representations are non-trivial
        mcfg = Config(cfg.to_dict())
        mcfg.multi_step.horizons = []
        mcfg.event_pred.enabled = False
        mcfg.vision.enabled = False
        model = WorldModel(schema, mcfg, num_actions=6).to(device)
        from open_player.training.trainer import WorldModelTrainer
        trainer = WorldModelTrainer(model, mcfg, schema, device=device)
        for i in range(min(train_steps, len(states) - 1)):
            trainer.online_step(states[i], rule.act(states[i]).index, states[i + 1], 0.0, False, 0.0)

        # probe data: per-slot features + semantic labels
        feats_pooled: List[np.ndarray] = []
        feats_per_entity: List[np.ndarray] = []
        labels: List[int] = []
        type_index = {"empty": 0, "player": 1, "enemy": 2, "resource": 3, "wall": 4}
        model.eval()
        with torch.no_grad():
            for st in states:
                rep = model.representation(st)
                pooled = rep.entity_emb[0].detach().cpu().numpy()  # [H]
                per = model.representation.entity_mlp(torch.cat([st.entities_t, st.beliefs_t], dim=-1))[0].detach().cpu().numpy()  # [N, H]
                for i, etype in enumerate(st.semantic_types):
                    if etype in type_index:
                        feats_pooled.append(pooled)
                        feats_per_entity.append(per[i])
                        labels.append(type_index[etype])
        # linear probe accuracy (70/30 split)
        accs = {}
        for name, feats in (("pooled", feats_pooled), ("per_entity", feats_per_entity)):
            X = torch.tensor(np.stack(feats), dtype=torch.float32, device=device)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            perm = torch.randperm(X.shape[0], device=device)
            cut = int(X.shape[0] * 0.7)
            tr, te = perm[:cut], perm[cut:]
            probe = torch.nn.Linear(X.shape[1], len(type_index)).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
            for _ in range(400):
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(probe(X[tr]), y[tr])
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred = probe(X[te]).argmax(-1)
                accs[name] = float((pred == y[te]).float().mean())
        rows.append({"seed": seed, "pooled_probe_acc": accs["pooled"], "per_entity_probe_acc": accs["per_entity"]})
        print(f"[repr] seed={seed} {accs}", flush=True)
    save_csv(run_dir, "representation_ablation.csv", rows)
    save_json(run_dir, "representation_ablation.json", rows)
    summary = aggregate_rows(rows)
    save_json(run_dir, "representation_ablation_summary.json", summary)
    stats = {}
    pooled = [r["pooled_probe_acc"] for r in rows]
    per_ent = [r["per_entity_probe_acc"] for r in rows]
    if len(pooled) >= 2:
        stats["per_entity_vs_pooled"] = is_improvement(per_ent, pooled, higher_is_better=True)
    save_json(run_dir, "representation_ablation_stats.json", stats)
    return stats


# --------------------------------------------------------------------------- #
# E11/E12: transfer benchmark + generalization groups
# --------------------------------------------------------------------------- #
def run_transfer_benchmark(
    cfg: Config,
    pair: Any,
    protocol: EvalProtocol,
    seeds: List[int],
    adaptation_steps: List[int],
    ckpt_dir: str,
    base_step: int,
    run_dir: str,
    worlds: Optional[List[str]] = None,
    agents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    world_set = set(worlds) if worlds else {"world_b", "world_c"}
    for world_name, env in (("world_b", pair.world_b), ("world_c", pair.world_c)):
        if env is None or world_name not in world_set:
            continue
        summary = run_adaptation(cfg, env, protocol, seeds, adaptation_steps, ckpt_dir, base_step, run_dir, agents=agents, world_name=world_name)
        out[world_name] = summary
        save_json(run_dir, f"transfer_{world_name}.json", summary)
    save_json(run_dir, "transfer_benchmark.json", out)
    return out


def run_generalization(
    cfg: Config,
    protocol: EvalProtocol,
    seeds: List[int],
    ckpt_dir: str,
    base_step: int,
    run_dir: str,
) -> Dict[str, Any]:
    from open_player.environments.transfer import make_env_variant
    gen = cfg.get("validation.generalization", {})
    groups = {}
    speeds = gen.get("enemy_speeds", {})
    for name, sp in speeds.items():
        groups[f"speed_{name}"] = {"enemy_move_prob": float(sp)}
    dens = gen.get("resource_densities", {})
    for name, d in dens.items():
        groups[f"density_{name}"] = {"num_resources": int(d)}
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        path = _ckpt_path(ckpt_dir, seed, base_step)
        if not os.path.exists(path):
            continue
        p1 = new_player(cfg, "side")
        p1.load_checkpoint(path)
        p0 = new_player(cfg, "structured")
        for gname, overrides in groups.items():
            env = make_env_variant(cfg, overrides)
            for agent, player in (("phase0", p0), ("phase1", p1)):
                s = player.evaluate(env, episodes=protocol.episodes, max_steps=protocol.max_steps, seed=seed)
                rows.append({
                    "seed": seed, "group": gname, "agent": agent,
                    "collected": s.get("mean_collected", 0.0),
                    "coverage": s.get("mean_exploration_coverage", 0.0),
                    "goal_success_rate": s.get("goal_success_rate", 0.0),
                })
    save_csv(run_dir, "generalization.csv", rows)
    save_json(run_dir, "generalization.json", rows)
    summary = {}
    for gname in groups:
        for agent in ("phase0", "phase1"):
            sub = [r for r in rows if r["group"] == gname and r["agent"] == agent]
            if sub:
                summary[f"{gname}.{agent}"] = aggregate_rows(sub)
    save_json(run_dir, "generalization_summary.json", summary)
    return summary
