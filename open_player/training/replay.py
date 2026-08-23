"""Replay buffer for the self-supervised world learning loop."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from open_player.core.schema import SchemaSet
from open_player.core.state import stack_world_states
from open_player.core.types import WorldState


@dataclass
class Transition:
    """One stored transition: (WorldState_t, action, WorldState_t+1, ...)."""

    state: WorldState  # compacted (float16 spatial/relations) for storage
    action: int
    next_state: WorldState
    reward: float
    done: bool
    change: float  # 1.0 if a semantic event happened on this step


class ReplayBuffer:
    """Bounded FIFO replay of transitions."""

    def __init__(self, capacity: int = 2048, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self._buffer: deque[Transition] = deque(maxlen=self.capacity)
        self._rng = torch.Generator().manual_seed(int(seed))

    def store(
        self,
        state: WorldState,
        action: int,
        next_state: WorldState,
        reward: float,
        done: bool,
        change: float,
    ) -> None:
        self._buffer.append(
            Transition(
                state=state.compact(),
                action=int(action),
                next_state=next_state.compact(),
                reward=float(reward),
                done=bool(done),
                change=float(change),
            )
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self._buffer) >= batch_size

    def sample(self, batch_size: int, schema: SchemaSet, device: Any = "cpu") -> Dict[str, Any]:
        """Sample a batch: batched WorldState pair + action/reward/done/change."""
        if not self.is_ready(batch_size):
            raise RuntimeError(f"replay buffer has {len(self)} transitions, need {batch_size}")
        idx = torch.randint(0, len(self._buffer), (batch_size,), generator=self._rng).tolist()
        transitions = [self._buffer[i] for i in idx]
        state = stack_world_states(schema, [t.state for t in transitions], device=device)
        next_state = stack_world_states(schema, [t.next_state for t in transitions], device=device)
        return {
            "state": state,
            "next_state": next_state,
            "action": torch.tensor([t.action for t in transitions], dtype=torch.long, device=device),
            "reward": torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=device),
            "done": torch.tensor([float(t.done) for t in transitions], dtype=torch.float32, device=device),
            "change": torch.tensor([t.change for t in transitions], dtype=torch.float32, device=device),
        }

    def clear(self) -> None:
        self._buffer.clear()

    def stats(self) -> Dict[str, Any]:
        if not self._buffer:
            return {"len": 0, "mean_reward": 0.0, "change_ratio": 0.0}
        rewards = [t.reward for t in self._buffer]
        changes = [t.change for t in self._buffer]
        return {
            "len": len(self._buffer),
            "mean_reward": sum(rewards) / len(rewards),
            "change_ratio": sum(changes) / len(changes),
        }
