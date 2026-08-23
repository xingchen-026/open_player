"""WorkingMemory: recent events, states, actions and the current goal."""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Optional

from open_player.core.types import Event, Goal, WorldState


class WorkingMemory:
    """Fixed-capacity short-term buffer used by the planner and the agent."""

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = int(capacity)
        self.recent_events: Deque[Event] = deque(maxlen=self.capacity)
        self.recent_states: Deque[WorldState] = deque(maxlen=8)
        self.recent_actions: Deque[int] = deque(maxlen=self.capacity)
        self.current_goal: Optional[Goal] = None
        self.last_reward: float = 0.0

    def add_event(self, event: Event) -> None:
        self.recent_events.append(event)

    def add_state(self, state: WorldState) -> None:
        self.recent_states.append(state.detach().to("cpu"))

    def add_action(self, action: int) -> None:
        self.recent_actions.append(action)

    def set_goal(self, goal: Optional[Goal]) -> None:
        self.current_goal = goal

    def latest_state(self) -> Optional[WorldState]:
        return self.recent_states[-1] if self.recent_states else None

    def clear(self) -> None:
        self.recent_events.clear()
        self.recent_states.clear()
        self.recent_actions.clear()
        self.current_goal = None
        self.last_reward = 0.0

    def snapshot(self) -> dict:
        return {
            "num_events": len(self.recent_events),
            "num_states": len(self.recent_states),
            "num_actions": len(self.recent_actions),
            "goal": self.current_goal.goal_id if self.current_goal else None,
            "last_reward": self.last_reward,
        }
