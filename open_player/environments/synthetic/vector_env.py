"""Vectorized synthetic environment: N parallel worlds, single process.

Phase 1 uses this for batched evaluation episodes and skill data collection.
The simulation itself stays in plain Python per world (simple, no
multiprocessing); the benefit is that downstream model work can batch N
transitions on the GPU.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from open_player.actions.specs import DiscreteActionSpace
from open_player.core.types import Action, Observation
from open_player.environments.synthetic.env import SyntheticGridEnv


class SyntheticGridVecEnv:
    """N independent SyntheticGridEnv instances stepped in one call."""

    def __init__(self, config: Any, n_envs: int = 8, seed: int = 0) -> None:
        self.config = config
        self.n_envs = int(n_envs)
        self.seed = int(seed)
        self.envs: List[SyntheticGridEnv] = []
        for i in range(self.n_envs):
            sub = config.merge({"seed": int(seed) + i})
            self.envs.append(SyntheticGridEnv(sub))
        self.action_space: DiscreteActionSpace = self.envs[0].action_space

    def reset(self, seeds: Optional[Sequence[int]] = None) -> List[Observation]:
        obs: List[Observation] = []
        for i, env in enumerate(self.envs):
            s = None if seeds is None else int(seeds[i])
            obs.append(env.reset(seed=s))
        return obs

    def step(self, actions: Sequence[int]) -> Tuple[List[Observation], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step every env; returns (obs, rewards [N], dones [N], infos)."""
        obs: List[Observation] = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos: List[Dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            a = int(actions[i])
            o, r, d, info = env.step(a)
            obs.append(o)
            rewards[i] = r
            dones[i] = d
            infos.append(info)
        return obs, rewards, dones, infos

    def step_actions(self, actions: Sequence[Action]) -> Tuple[List[Observation], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        return self.step([a.index for a in actions])

    @property
    def action_names(self) -> List[str]:
        return list(self.action_space.names)
