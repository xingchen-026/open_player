"""SkillTrainer: behavior cloning from successful rule trajectories.

    RuleSkill -> collect trajectories -> successful trajectory filtering
    -> supervised skill learning -> NeuralSkill -> evaluation

No policy gradient in the first version: BC gives a stable NeuralSkill, and
the interface leaves room for intrinsic-reward fine-tuning later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from open_player.core.types import Action, Observation, WorldState
from open_player.skills.neural import NeuralSkill, StateFeaturizer


@dataclass
class SkillTrainReport:
    """Serialisable BC training report."""

    skill_name: str = ""
    data_steps: int = 0
    kept_episodes: int = 0
    epochs: int = 0
    final_loss: float = 0.0
    action_accuracy: float = 0.0
    termination_accuracy: float = 0.0
    params: int = 0
    save_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "data_steps": self.data_steps,
            "kept_episodes": self.kept_episodes,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "action_accuracy": self.action_accuracy,
            "termination_accuracy": self.termination_accuracy,
            "params": self.params,
            "save_path": self.save_path,
            "extra": dict(self.extra),
        }


class SkillTrainer:
    """Collects rule trajectories and trains a NeuralSkill by cloning them."""

    def __init__(self, config: Any, featurizer: StateFeaturizer, device: Any = "cpu") -> None:
        self.config = config
        self.featurizer = featurizer
        self.device = device
        sc = config.skill_training if hasattr(config, "skill_training") else config.get("skill_training", {})
        self.epochs = int(sc.get("bc_epochs", 15))
        self.batch_size = int(sc.get("bc_batch_size", 64))
        self.lr = float(sc.get("lr", 0.001))
        self.min_steps = int(sc.get("min_trajectory_steps", 8))
        self.min_coverage = float(sc.get("success_min_coverage", 0.15))
        self.need_goal = bool(sc.get("success_goal_succeeded", True))
        self.skill_name = str(sc.get("skill_name", "neural_explore"))

    # ------------------------------------------------------------------ #
    def collect(
        self,
        env: Any,
        perceive: Callable[[Observation, int], WorldState],
        policy: Callable[[WorldState], Action],
        steps: int,
        seed: int = 0,
    ) -> Tuple[List[np.ndarray], List[int], List[int], Dict[str, Any]]:
        """Collect (state features, action, termination) from kept episodes.

        A trajectory (env episode) is kept when it reached the exploration
        coverage threshold or completed a goal (config-driven).
        """
        xs: List[np.ndarray] = []
        actions: List[int] = []
        terms: List[int] = []
        kept = 0
        done_steps = 0
        obs = env.reset(seed=seed)
        state = perceive(obs, 0)
        traj: List[Tuple[np.ndarray, int]] = []
        while done_steps < steps:
            action = policy(state)
            obs2, reward, done, info = env.step(action)
            state2 = perceive(obs2, env.world.steps)
            feats = self.featurizer.features(state)[0].detach().cpu().numpy().astype(np.float32)
            traj.append((feats, action.index))
            done_steps += 1
            if done:
                w = env.world
                free = w.grid_size * w.grid_size - len(w.walls)
                coverage = len(w.visited) / max(free, 1)
                keep = (len(traj) >= self.min_steps) and (coverage >= self.min_coverage or (self.need_goal and info.get("collected", 0) >= 1))
                if keep:
                    kept += 1
                    for i, (f, a) in enumerate(traj):
                        xs.append(f)
                        actions.append(a)
                        terms.append(1 if i == len(traj) - 1 else 0)
                traj = []
                obs = env.reset()
                state = perceive(obs, done_steps)
            else:
                state = state2
        if traj and len(traj) >= self.min_steps:
            kept += 1
            for i, (f, a) in enumerate(traj):
                xs.append(f)
                actions.append(a)
                terms.append(1 if i == len(traj) - 1 else 0)
        return xs, actions, terms, {"data_steps": done_steps, "kept_episodes": kept, "pairs": len(xs)}

    # ------------------------------------------------------------------ #
    def train(
        self,
        skill: NeuralSkill,
        xs: List[np.ndarray],
        actions: List[int],
        terms: List[int],
    ) -> SkillTrainReport:
        """Supervised (BC) training: CE on actions + BCE on termination."""
        if len(xs) < self.batch_size:
            raise RuntimeError(f"not enough BC data ({len(xs)} pairs < batch {self.batch_size})")
        X = torch.from_numpy(np.stack(xs)).to(self.device)
        A = torch.tensor(actions, dtype=torch.long, device=self.device)
        T = torch.tensor(terms, dtype=torch.float32, device=self.device).unsqueeze(1)
        opt = torch.optim.Adam(skill.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        bce = nn.BCEWithLogitsLoss()
        n = X.shape[0]
        final_loss = 0.0
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                logits, term_logit = skill(X[idx])
                loss = ce(logits, A[idx]) + 0.5 * bce(term_logit, T[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                final_loss = float(loss.detach().cpu())
        skill.eval()
        with torch.no_grad():
            logits, term_logit = skill(X)
            acc = float((logits.argmax(-1) == A).float().mean())
            tacc = float(((torch.sigmoid(term_logit) > 0.5).float() == T).float().mean())
        return SkillTrainReport(
            skill_name=skill.name,
            data_steps=len(xs),
            kept_episodes=0,
            epochs=self.epochs,
            final_loss=final_loss,
            action_accuracy=acc,
            termination_accuracy=tacc,
            params=skill.num_parameters(),
        )

    def save(self, skill: NeuralSkill, path: str) -> str:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"skill_state": skill.state_dict(), "action_names": skill.action_names, "skill_name": skill.name}, path)
        return path

    def load(self, skill: NeuralSkill, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        skill.load_state_dict(payload["skill_state"])
