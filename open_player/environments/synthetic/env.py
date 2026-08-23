"""SyntheticGridEnv: the Phase 0 environment (Gym-style interface)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from open_player.actions.specs import DiscreteActionSpace
from open_player.core.types import Action, Observation
from open_player.environments.base import Environment
from open_player.environments.synthetic.world import GridWorld

DEFAULT_ACTION_NAMES = ("noop", "up", "down", "left", "right", "collect")


class SyntheticGridEnv(Environment):
    """2D grid world with player / enemies / resources / walls / fog."""

    def __init__(self, config: Any) -> None:
        ec = config.environment
        self.config = config
        self._action_space = DiscreteActionSpace(list(DEFAULT_ACTION_NAMES))
        self.max_steps = int(ec.max_steps)
        self._world = GridWorld(
            grid_size=int(ec.grid_size),
            num_enemies=int(ec.num_enemies),
            num_resources=int(ec.num_resources),
            fog_radius=int(ec.fog_radius),
            player_hp=int(ec.player_hp),
            enemy_move_prob=float(ec.enemy_move_prob),
            enemy_attack_prob=float(ec.get("enemy_attack_prob", 0.5)),
            seed=int(config.seed),
        )
        self._reward_cfg = ec.reward
        self._t = 0

    # -- Environment interface ------------------------------------------- #
    def reset(self, seed: Optional[int] = None) -> Observation:
        self._world.reset(seed=seed)
        self._t = 0
        return self._world.build_observation(t=0)

    def step(self, action: Action | int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        name = self._resolve_action(action)
        reward, done, info = self._world.step(name)
        self._t += 1
        timed_out = self._t >= self.max_steps
        done = bool(done) or timed_out
        info["env_t"] = self._t
        info["timeout"] = timed_out
        obs = self._world.build_observation(t=self._t)
        return obs, reward, done, info

    # -- helpers ---------------------------------------------------------- #
    def _resolve_action(self, action: Action | int) -> str:
        if isinstance(action, Action):
            idx = action.index
        else:
            idx = int(action)
        return self.action_space.name(idx)

    def render_ascii(self) -> str:
        from open_player.environments.synthetic.renderer import AsciiRenderer
        return AsciiRenderer().render_env(self)

    @property
    def world(self) -> GridWorld:
        return self._world
