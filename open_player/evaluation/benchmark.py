"""Phase 1 evaluation primitives: agent/baseline/world-model benchmarks."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch

from open_player.core.types import Action, Observation, WorldState
from open_player.evaluation.baselines import RandomBaseline, RuleBaseline
from open_player.evaluation.metrics import episode_metrics, summarize_metrics


def evaluate_world_model(
    model: Any,
    perceive: Callable[[Observation, int], WorldState],
    env: Any,
    steps: int = 8,
    seed: int = 0,
) -> Dict[str, float]:
    """1/4/8-step prediction errors on one fresh env sequence."""
    model.eval()
    obs = env.reset(seed=seed)
    state = perceive(obs, 0)
    states: List[WorldState] = [state]
    actions: List[int] = []
    for i in range(steps):
        a = i % env.action_space.n
        obs2, _r, _d, _i = env.step(a)
        states.append(perceive(obs2, i + 1))
        actions.append(a)
    with torch.no_grad():
        errors = model.prediction_errors(states[0], actions, states[1:])
    return errors


def evaluate_baseline(
    env: Any,
    kind: str,
    perceive: Callable[[Observation, int], WorldState],
    episodes: int,
    max_steps: int,
    schema: Any,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the random / rule baseline for several episodes."""
    if kind == "random":
        policy = RandomBaseline(env.action_space, seed=seed)
    elif kind == "rule":
        policy = RuleBaseline(env.action_space, schema, seed=seed)
    else:
        raise ValueError(f"unknown baseline kind '{kind}'")
    entries: List[Dict[str, Any]] = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + 100 * ep)
        state = perceive(obs, 0)
        total_reward = 0.0
        info: Dict[str, Any] = {}
        for t in range(max_steps):
            action = policy.act(state)
            obs2, reward, done, info = env.step(action)
            total_reward += float(reward)
            state = perceive(obs2, t + 1)
            if done:
                break
        entries.append(episode_metrics(env, info, total_reward, t + 1))
    summary = summarize_metrics(entries)
    summary["kind"] = kind
    summary["episodes"] = episodes
    return summary


def evaluate_agent(
    player: Any,
    env: Any,
    episodes: int,
    max_steps: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate a Player (no weight updates) over several episodes."""
    return player.evaluate(env, episodes=episodes, max_steps=max_steps, seed=seed)
