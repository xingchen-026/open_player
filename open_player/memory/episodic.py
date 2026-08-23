"""EpisodicMemory + EpisodeBuilder (event-boundary + goal hybrid)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from open_player.core.types import Episode, EpisodeOutcome, Event, Goal, WorldState


class EpisodeBuilder:
    """Builds one Episode incrementally; a boundary closes it.

    Boundaries in Phase 0: goal completed/failed or the environment episode
    ended (event-boundary + goal hybrid).
    """

    def __init__(self, episode_id: str, goal: Optional[Goal], state: Optional[WorldState], t: int) -> None:
        self.episode = Episode(
            episode_id=episode_id,
            goal=goal,
            start_state=None if state is None else state.detach().to("cpu"),
            start_t=int(t),
        )

    def on_event(self, event: Event) -> None:
        self.episode.events.append(event)

    def on_skill(self, skill_name: str) -> None:
        if not self.episode.skill_trace or self.episode.skill_trace[-1] != skill_name:
            self.episode.skill_trace.append(skill_name)

    def on_action(self, action: int) -> None:
        self.episode.action_trace.append(int(action))

    def finish(
        self,
        outcome: EpisodeOutcome,
        state: Optional[WorldState],
        t: int,
        failure_analysis: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        self.episode.outcome = outcome
        self.episode.end_state = None if state is None else state.detach().to("cpu")
        self.episode.end_t = int(t)
        self.episode.failure_analysis = dict(failure_analysis or {})
        return self.episode


class EpisodicMemory:
    """Stores finished episodes; simple filter queries (no vector DB)."""

    def __init__(self, capacity: int = 100) -> None:
        self.capacity = int(capacity)
        self.episodes: List[Episode] = []

    def store(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity :]

    def query(
        self,
        goal_type: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 10,
    ) -> List[Episode]:
        out = list(self.episodes)
        if goal_type is not None:
            out = [e for e in out if e.goal is not None and e.goal.goal_type == goal_type]
        if outcome is not None:
            out = [e for e in out if e.outcome.value == outcome]
        return out[-limit:]

    def stats(self) -> Dict[str, Any]:
        if not self.episodes:
            return {"num_episodes": 0}
        from collections import Counter
        outcomes = Counter(e.outcome.value for e in self.episodes)
        goal_types = Counter(e.goal.goal_type if e.goal else "none" for e in self.episodes)
        return {
            "num_episodes": len(self.episodes),
            "outcomes": dict(outcomes),
            "goal_types": dict(goal_types),
        }

    def __len__(self) -> int:
        return len(self.episodes)
