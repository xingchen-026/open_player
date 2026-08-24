"""Configuration, seeds, device resolution and logging helpers.

Engineering rules enforced here:

* every hyperparameter lives in a config (YAML file or dict), never in code;
* deterministic seeds are one call away;
* CUDA is auto-detected (device: auto), CPU always works.
"""
from __future__ import annotations

import copy
import logging
import os
import random
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import yaml

_LOGGING_CONFIGURED = False

_DEFAULT_CONFIG: Dict[str, Any] = {
    "meta": {"name": "open-player-phase0", "version": 1},
    "seed": 0,
    "device": "auto",
    "logging": {"level": "INFO"},
    "schema": {
        "max_entities": 32,
        "world_size": 12,
        "max_speed": 1.0,
        "max_size": 3.0,
        "entity_fields": [
            {"name": "position", "dim": 2},
            {"name": "velocity", "dim": 2},
            {"name": "size", "dim": 1},
            {"name": "appearance", "dim": 8},
            {"name": "semantic_features", "dim": 16},
            {"name": "dynamics_features", "dim": 12},
            {"name": "status", "dim": 1},
        ],
        "belief_dim": 8,
        "relation_fields": [
            {"name": "distance", "dim": 1},
            {"name": "direction", "dim": 2},
            {"name": "relative_velocity", "dim": 2},
            {"name": "overlap", "dim": 1},
            {"name": "visibility", "dim": 1},
            {"name": "semantic_relation", "dim": 1},
        ],
        "spatial": {"channels": 16, "height": 32, "width": 32},
        "dynamics_dim": 64,
        "temporal_dim": 16,
        "global_dim": 32,
        "uncertainty_dim": 8,
    },
    "environment": {
        "grid_size": 12,
        "num_enemies": 2,
        "num_resources": 4,
        "fog_radius": 5,
        "max_steps": 120,
        "player_hp": 3,
        "enemy_move_prob": 0.8,
        "enemy_attack_prob": 0.5,
        "reward": {"collect": 1.0, "explore": 0.02, "step": -0.01, "death": -1.0},
    },
    "world_model": {
        "entity_hidden": 128,
        "spatial_channels": [32, 64],
        "spatial_embed": 256,
        "relation_hidden": 32,
        "latent_dim": 64,
        "dynamics_hidden": 128,
        "head_hidden": 256,
        "action_embedding_dim": 32,
        "loss_weights": {"entity": 1.0, "spatial": 0.1, "change": 0.5, "latent": 0.1},
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 0.001,
        "replay_capacity": 2048,
        "update_every": 1,
        "replay_update_every": 8,
        "replay_updates_per_tick": 1,
        "exploration_eps": 0.15,
        "epsilon_decay": 0.999,
        "grad_clip": 1.0,
        "steps": 2000,
        "log_every": 100,
        "goal_min_exploration": 0.6,
    },
    "planning": {
        "horizons": {"short": 4, "medium": 8, "long": 32},
        "rollout_steps": 4,
        "max_candidates": 4,
        "goal_type_horizons": {
            "task": "short",
            "survival": "short",
            "skill_improvement": "short",
            "exploration": "medium",
            "information": "long",
            "learning": "long",
        },
    },
    "goals": {
        "novelty_threshold": 0.2,
        "threat_threshold": 0.5,
        "learning_threshold": 0.05,
        "type_weights": {
            "task": 1.0,
            "survival": 0.9,
            "exploration": 0.5,
            "information": 0.3,
            "learning": 0.3,
            "skill_improvement": 0.2,
        },
    },
    "memory": {"working_capacity": 64, "episodic_capacity": 100},
    "checkpoint": {"dir": "checkpoints", "keep_last": 2},
}


class Config:
    """Nested attribute-style configuration.

    Child Configs created through attribute access share the same underlying
    dict, so mutations like cfg.environment.player_hp = 10 persist.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, _shared: bool = False) -> None:
        if _shared:
            object.__setattr__(self, "_data", data if data is not None else {})
        else:
            object.__setattr__(self, "_data", copy.deepcopy(data or {}))

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name in data:
            value = data[name]
            return Config(value, _shared=True) if isinstance(value, dict) else value
        raise AttributeError(f"config has no key '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        data = object.__getattribute__(self, "_data")
        data[name] = value

    def __contains__(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_data")

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, "_data")[key]

    def __iter__(self) -> Iterator[str]:
        return iter(object.__getattribute__(self, "_data"))

    def keys(self) -> Any:
        return object.__getattribute__(self, "_data").keys()

    def get(self, path: str, default: Any = None) -> Any:
        """Dotted-path getter, e.g. cfg.get('training.batch_size')."""
        node: Any = object.__getattribute__(self, "_data")
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(object.__getattribute__(self, "_data"))

    def merge(self, other: Dict[str, Any]) -> "Config":
        """Shallow-merge extra keys into a new config (used by CLI overrides)."""
        merged = copy.deepcopy(object.__getattribute__(self, "_data"))

        def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _merge(dst[k], v)
                else:
                    dst[k] = copy.deepcopy(v)

        _merge(merged, other)
        return Config(merged)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self.to_dict()})"


def load_config(path: str) -> Config:
    """Load a YAML config file into a Config object."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data)


def default_config() -> Config:
    """Embedded default configuration (mirrors configs/phase0.yaml)."""
    return Config(_DEFAULT_CONFIG)


_PHASE1_DEFAULTS: Dict[str, Any] = {
    "environment": {
        "grid_size": 12,
        "num_enemies": 2,
        "num_resources": 4,
        "fog_radius": 5,
        "max_steps": 120,
        "player_hp": 4,
        "enemy_move_prob": 0.6,
        "enemy_attack_prob": 0.4,
        "wall_density": 0.05,
        "resource_cluster": True,
        "render_rgb": True,
    },
    "vision": {
        "enabled": True,
        "input_height": 90,
        "input_width": 160,
        "channels": [32, 64, 128, 256],
        "spatial_channels": 16,
        "patch_hidden": 256,
        "patch_out": 24,
        "detach_targets": True,
        "spatial_reg": 0.05,
    },
    "multi_step": {"horizons": [4, 8], "loss_weights": {"step1": 1.0, "step4": 0.5, "step8": 0.25}, "window": 8},
    "rollout_schedule": {
        "initial_teacher_forcing": 0.9,
        "final_teacher_forcing": 0.2,
        "anneal_steps": 5000,
        "rollout_ratio": 0.2,
    },
    "event_pred": {"enabled": True, "hidden": 64, "conf_blend": 0.5, "loss_weight": 1.0},
    "intrinsic": {
        "alpha": 1.0,
        "beta": 0.3,
        "gamma": 0.2,
        "risk_penalty": 0.5,
        "repetition_penalty": 0.2,
        "novelty_decay": 0.99,
        "visit_count_cap": 10,
        "info_goal_bonus": 0.3,
    },
    "skill_training": {
        "bc_epochs": 15,
        "bc_batch_size": 64,
        "lr": 0.001,
        "min_trajectory_steps": 8,
        "success_min_coverage": 0.15,
        "success_goal_succeeded": True,
        "train_steps": 400,
        "skill_name": "neural_explore",
    },
    "evaluation": {
        "eval_interval": 500,
        "eval_episodes": 4,
        "eval_max_steps": 100,
        "curve_steps": [1000, 5000, 10000, 20000],
        "adaptation_steps": 1000,
        "log_dir": "runs/phase1",
    },
    "transfer": {
        "world_a": {
            "name": "world_a_open",
            "grid_size": 12,
            "num_enemies": 1,
            "num_resources": 4,
            "fog_radius": 6,
            "max_steps": 120,
            "player_hp": 6,
            "enemy_move_prob": 0.3,
            "enemy_attack_prob": 0.3,
            "wall_density": 0.03,
            "resource_cluster": True,
            "render_rgb": True,
        },
        "world_b": {
            "name": "world_b_narrow",
            "grid_size": 14,
            "num_enemies": 2,
            "num_resources": 6,
            "fog_radius": 4,
            "max_steps": 150,
            "player_hp": 6,
            "enemy_move_prob": 0.7,
            "enemy_attack_prob": 0.6,
            "wall_density": 0.16,
            "resource_cluster": False,
            "render_rgb": True,
        },
    },
}


def phase1_config() -> Config:
    """Phase 0 defaults + Phase 1 sections (mirrors configs/phase1.yaml)."""
    return Config(_DEFAULT_CONFIG).merge(_PHASE1_DEFAULTS)


def resolve_device(cfg: Config) -> torch.device:
    """Resolve cfg.device ('auto' -> cuda if available, else cpu)."""
    spec = cfg.get("device", "auto")
    if spec in (None, "", "auto"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(spec))
    return device


def set_seed(seed: int) -> None:
    """Set every random source the project uses (deterministic experiments)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(level: str = "INFO", name: str = "open_player") -> logging.Logger:
    """Idempotent logging setup; returns the project logger."""
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, str(level).upper(), logging.INFO),
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOGGING_CONFIGURED = True
    return logging.getLogger(name)
