"""Player: the coordinating agent of Open Player Phase 0.

The Player wires Observation -> WorldState -> WorldModel -> Prediction ->
Loss -> Backprop together with Events -> Episodes -> Goals -> Planner ->
Skills -> Actions, but it never reimplements any module's logic.

Public API (frozen direction):

    player.learn(environment)   # run + train the world model
    player.run(environment)     # act with the trained model (no weight updates)

Internals remain accessible per the API design principles:

    player.world_model.predict(...)
    player.memory.store(...)
    player.planner.plan(...)
    player.skills.act(...)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from open_player.actions.controller import ActionController
from open_player.actions.specs import DiscreteActionSpace
from open_player.core.config import Config, resolve_device, set_seed
from open_player.core.schema import SchemaSet
from open_player.core.specs import Environment
from open_player.core.types import (
    Action,
    EpisodeOutcome,
    Event,
    Goal,
    Observation,
    SEMANTIC_EVENT_TYPES,
    WorldState,
)
from open_player.environments.synthetic.env import DEFAULT_ACTION_NAMES
from open_player.events.detector import HeuristicEventDetector
from open_player.events.graph import EventGraph
from open_player.memory.episodic import EpisodicMemory, EpisodeBuilder
from open_player.memory.procedural import ProceduralMemory
from open_player.memory.semantic import SemanticMemory
from open_player.memory.spatial import SpatialMemoryStore
from open_player.memory.working import WorkingMemory
from open_player.motivation.goals import GoalManager
from open_player.motivation.motivation import IntrinsicMotivation
from open_player.observation.dummy import DummyVisionEncoder
from open_player.planning.planner import Plan, Planner
from open_player.skills.registry import SkillRegistry
from open_player.tracking.tracker import BeliefTracker
from open_player.training.trainer import WorldModelTrainer
from open_player.world.model import WorldModel

log = logging.getLogger("open_player.agent")


@dataclass
class PlayerReport:
    """Serialisable result of learn()/run()."""

    total_steps: int = 0
    episodes: int = 0
    events: int = 0
    goals_succeeded: int = 0
    collected_total: int = 0
    final_loss: Dict[str, float] = field(default_factory=dict)
    mean_reward: float = 0.0
    checkpoint: Optional[str] = None
    wall_time_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "events": self.events,
            "goals_succeeded": self.goals_succeeded,
            "collected_total": self.collected_total,
            "final_loss": dict(self.final_loss),
            "mean_reward": self.mean_reward,
            "checkpoint": self.checkpoint,
            "wall_time_s": self.wall_time_s,
            "extra": dict(self.extra),
        }


class Player:
    """Phase 0 Learning Agent Core (coordinator)."""

    def __init__(self, config: Config, device: Any = None) -> None:
        self.config = config
        self.device = device if device is not None else resolve_device(config)
        set_seed(int(config.seed))
        self.rng = np.random.default_rng(int(config.seed))

        self.schema = SchemaSet.from_config(config)
        self.encoder = DummyVisionEncoder(self.schema, device=self.device)
        self.tracker = BeliefTracker(self.schema, device=self.device)

        # action / skill wiring (env-specific controller attached in attach())
        self.controller = ActionController(DiscreteActionSpace(list(DEFAULT_ACTION_NAMES)))
        self.registry = SkillRegistry.build_default(self.controller, config)

        # world model + trainer
        self.world_model = WorldModel(self.schema, config, num_actions=len(self.controller.space.names))
        self.world_model.to(self.device)
        self.trainer = WorldModelTrainer(self.world_model, config, self.schema, device=self.device)

        # cognition
        self.detector = HeuristicEventDetector()
        self.event_graph = EventGraph()
        self.working = WorkingMemory(capacity=int(config.get("memory.working_capacity", 64)))
        self.episodic = EpisodicMemory(capacity=int(config.get("memory.episodic_capacity", 100)))
        self.procedural = ProceduralMemory()
        self.semantic = SemanticMemory()
        self.spatial_store = SpatialMemoryStore(self.schema.spatial.shape, device=self.device)
        self.motivation = IntrinsicMotivation(config)
        self.goal_manager = GoalManager(config)
        self.planner = Planner(config, self.registry, self.schema, model=self.world_model, procedural_memory=self.procedural)

        # running state
        self.env: Optional[Environment] = None
        self.state: Optional[WorldState] = None
        self.goal: Optional[Goal] = None
        self.plan: Optional[Plan] = None
        self.episode_builder: Optional[EpisodeBuilder] = None
        self._episode_counter = 0
        self._goal_counter = 0
        self._collected_total = 0
        self._last_error: Optional[float] = None
        self._steps_since_plan = 0
        self.eps = float(config.get("training.exploration_eps", 0.15))
        self.eps_decay = float(config.get("training.epsilon_decay", 0.999))

    # ------------------------------------------------------------------ #
    # wiring
    # ------------------------------------------------------------------ #
    def attach(self, environment: Environment) -> None:
        """Bind an environment; rebuild the action-dependent parts."""
        self.env = environment
        names = environment.action_space.names
        self.controller = ActionController(DiscreteActionSpace(list(names)))
        self.registry = SkillRegistry.build_default(self.controller, self.config)
        if self.world_model.num_actions != len(names):
            self.world_model = WorldModel(self.schema, self.config, num_actions=len(names))
            self.world_model.to(self.device)
            self.trainer = WorldModelTrainer(self.world_model, self.config, self.schema, device=self.device)
        self.planner = Planner(self.config, self.registry, self.schema, model=self.world_model, procedural_memory=self.procedural)

    def reset_cognition(self) -> None:
        """Clear per-run cognitive state (episode/goal caches)."""
        self.working.clear()
        self.motivation.reset()
        self.spatial_store.reset()
        self._episode_counter = 0
        self._goal_counter = 0
        self._collected_total = 0
        self._last_error = None
        self._goal_report = None
        self.eps = float(self.config.get("training.exploration_eps", 0.15))

    # ------------------------------------------------------------------ #
    # perception / decision
    # ------------------------------------------------------------------ #
    def perceive(self, observation: Observation, t: int = 0) -> WorldState:
        """Observation (+ belief update) -> WorldState."""
        self.state = self.tracker.track(self.state, observation, t=t)
        return self.state

    def choose_action(self, state: WorldState, info: Dict[str, Any], explore: bool) -> Action:
        """Planner-driven action selection (with optional exploration)."""
        if self.goal is None or self.goal.status in ("succeeded", "failed", "abandoned"):
            self.goal = self.goal_manager.select(state, self._drives(state), info)
            self._goal_counter += 1
            self.plan = None
            if self.episode_builder is not None and self.episode_builder.episode.goal is None:
                self.episode_builder.episode.goal = self.goal
        elif self.goal.status == "active":
            # opportunistic re-scoring: switch when a candidate clearly wins
            drives = self._drives(state)
            alt = self.goal_manager.select(state, drives, info)
            cur_score = self.goal_manager.scorer.score(self.goal, state, drives)
            alt_score = self.goal_manager.scorer.score(alt, state, drives)
            if alt.goal_type != self.goal.goal_type and alt_score > cur_score + 0.3:
                self.goal.status = "abandoned"
                self.goal = alt
                self._goal_counter += 1
                self.plan = None
        if self.plan is None or self.plan.skill.should_terminate(state) or self._steps_since_plan >= self.plan.horizon:
            self.plan = self.planner.plan(state, self.goal)
            self._steps_since_plan = 0
        if explore and self.rng.random() < self.eps:
            action = self.controller.sample(self.rng)
        else:
            action = self.plan.skill.act(state, rng=self.rng)
        self._steps_since_plan += 1
        return action

    def _drives(self, state: WorldState) -> Dict[str, float]:
        return self.motivation.compute(state, world_model_error=self._last_error, reward=self.working.last_reward)

    # ------------------------------------------------------------------ #
    # main loops
    # ------------------------------------------------------------------ #
    def learn(
        self,
        environment: Environment,
        total_steps: Optional[int] = None,
        log_every: Optional[int] = None,
        checkpoint: Optional[str] = None,
        verbose: bool = True,
    ) -> PlayerReport:
        """Run the full closed loop while training the world model online."""
        self.attach(environment)
        self.reset_cognition()
        total_steps = total_steps or int(self.config.get("training.steps", 2000))
        log_every = log_every or int(self.config.get("training.log_every", 100))
        self.trainer.train()

        report = PlayerReport()
        self._goal_report = report
        t0 = time.time()
        obs = environment.reset(seed=int(self.config.seed))
        state = self.perceive(obs, t=0)
        self._start_episode(state, goal=None, t=0)
        self._update_spatial(state)
        env_info: Dict[str, Any] = {"threat_level": 0.0}
        total_reward = 0.0
        steps_in_env = 0

        for step in range(total_steps):
            action = self.choose_action(state, env_info, explore=True)
            obs2, reward, done, info = environment.step(action)
            total_reward += float(reward)
            state2 = self.perceive(obs2, t=step + 1)
            self._update_spatial(state2)

            # events -> graph, working memory, episode, semantic memory
            events = self.detector.detect(state, state2, info, step + 1)
            change = 1.0 if any(e.type in SEMANTIC_EVENT_TYPES for e in events) else 0.0
            self._record_events(events)

            # world model learning
            metrics = self.trainer.online_step(state, action.index, state2, float(reward), bool(done), change)
            if metrics:
                self._last_error = float(metrics.get("entity", self._last_error or 0.0))
            tick = self.trainer.tick()

            # motivation / goal bookkeeping
            self.working.last_reward = float(reward)
            self.working.add_state(state2)
            self.working.add_action(action.index)
            env_info = dict(info)
            env_info["world_model_error"] = self._last_error or 0.0
            env_info["num_resources"] = getattr(getattr(environment, "world", None), "num_resources", 1)
            self._track_goal(state2, env_info)

            steps_in_env += 1
            if done:
                report.episodes += 1
                self._finish_env_episode(state2, env_info, steps_in_env)
                steps_in_env = 0
                obs = environment.reset()
                state = self.perceive(obs, t=step + 1)
                self._start_episode(state, goal=self.goal, t=step + 1)
            else:
                state = state2

            self.eps = max(0.02, self.eps * self.eps_decay)
            if verbose and (step + 1) % log_every == 0:
                line = self._log_line(step + 1, total_steps, report, total_reward)
                log.debug(line)
                print(line, flush=True)

        report.total_steps = total_steps
        report.events = len(self.event_graph.nodes)
        report.collected_total = self._collected_total
        report.final_loss = dict(self.trainer.latest)
        report.mean_reward = total_reward / max(total_steps, 1)
        report.wall_time_s = time.time() - t0
        report.extra["episodic"] = self.episodic.stats()
        report.extra["event_graph"] = self.event_graph.stats()
        report.extra["procedural"] = self.procedural.stats()
        report.extra["replay"] = self.trainer.replay.stats()
        report.extra["model_params"] = self.world_model.num_parameters()
        if checkpoint:
            report.checkpoint = self.save_checkpoint(checkpoint)
        if verbose:
            print(self._summary_line(report), flush=True)
        return report

    def run(
        self,
        environment: Environment,
        max_steps: Optional[int] = None,
        render: bool = False,
        verbose: bool = True,
    ) -> PlayerReport:
        """Act in the environment with the current model (no training)."""
        self.attach(environment)
        self.trainer.eval()
        max_steps = max_steps or int(self.config.get("environment.max_steps", 120))
        report = PlayerReport()
        self._goal_report = report
        t0 = time.time()
        obs = environment.reset(seed=int(self.config.seed))
        state = self.perceive(obs, t=0)
        self._start_episode(state, goal=None, t=0)
        env_info: Dict[str, Any] = {"threat_level": 0.0}
        total_reward = 0.0
        if render and verbose:
            print(environment.render_ascii(), flush=True)

        for step in range(max_steps):
            action = self.choose_action(state, env_info, explore=False)
            obs2, reward, done, info = environment.step(action)
            total_reward += float(reward)
            state2 = self.perceive(obs2, t=step + 1)
            self._update_spatial(state2)
            events = self.detector.detect(state, state2, info, step + 1)
            self._record_events(events)
            self.working.last_reward = float(reward)
            env_info = dict(info)
            env_info["num_resources"] = getattr(getattr(environment, "world", None), "num_resources", 1)
            env_info["world_model_error"] = self._last_error or 0.0
            self._track_goal(state2, env_info)
            if verbose:
                try:
                    w = environment.world
                    pos = f" p={w.player_pos.tolist()} e={[e.position.tolist() for e in w.enemies if e.alive]} hp={w.player_hp}"
                except Exception:
                    pos = ""
                print(f"[run] t={step + 1} action={action.name:<8} reward={reward:+.2f} "
                      f"goal={self.goal.goal_type if self.goal else '-'} skill={self.plan.skill_name if self.plan else '-'} "
                      f"events={[e.type for e in events]}{pos}", flush=True)
            if render and verbose:
                print(environment.render_ascii(), flush=True)
            state = state2
            if done:
                report.episodes += 1
                self._finish_env_episode(state2, env_info, step + 1)
                break

        report.total_steps = step + 1
        report.events = len(self.event_graph.nodes)
        report.collected_total = self._collected_total
        report.mean_reward = total_reward / max(report.total_steps, 1)
        report.wall_time_s = time.time() - t0
        report.extra["goals"] = [self.goal.goal_type if self.goal else None]
        report.extra["goal_status"] = self.goal.status if self.goal else None
        if verbose:
            print(self._summary_line(report), flush=True)
        return report

    # ------------------------------------------------------------------ #
    # episode / goal lifecycle
    # ------------------------------------------------------------------ #
    def _start_episode(self, state: WorldState, goal: Optional[Goal], t: int) -> None:
        self._episode_counter += 1
        self.episode_builder = EpisodeBuilder(f"ep-{self._episode_counter}", goal, state, t)

    def _finish_env_episode(self, state: WorldState, env_info: Dict[str, Any], t: int) -> None:
        if self.episode_builder is None:
            return
        if env_info.get("death") or env_info.get("hp", 1) <= 0:
            outcome = EpisodeOutcome.FAILURE
            failure = {"reason": "death"}
        elif self.goal is not None and self.goal.status == "succeeded":
            outcome = EpisodeOutcome.SUCCESS
            failure = {}
        else:
            outcome = EpisodeOutcome.TIMEOUT
            failure = {"reason": "max_steps"}
        episode = self.episode_builder.finish(outcome, state, t, failure)
        self.episodic.store(episode)

    def _track_goal(self, state: WorldState, env_info: Dict[str, Any]) -> None:
        if self.goal is None:
            return
        status = self.goal_manager.update(self.goal, state, env_info)
        if env_info.get("collected_this_step"):
            self._collected_total += 1
            self.semantic.observe("player", "collected", str(env_info.get("collected_entity", "resource")))
        if status in ("succeeded", "failed"):
            report = getattr(self, "_goal_report", None)
            if report is not None:
                report.goals_succeeded += 1 if status == "succeeded" else 0
            if self.episode_builder is not None:
                outcome = EpisodeOutcome.SUCCESS if status == "succeeded" else EpisodeOutcome.FAILURE
                episode = self.episode_builder.finish(outcome, state, state.t, {} if status == "succeeded" else {"reason": "goal_failed"})
                self.episodic.store(episode)
                self._episode_counter += 1
                self.episode_builder = EpisodeBuilder(f"ep-{self._episode_counter}", None, state, state.t)
            # record skill outcome into procedural memory
            if self.plan is not None:
                self.procedural.record(self.plan.skill_name, success=(status == "succeeded"), reward=self.working.last_reward)
            self.goal = None
            self.plan = None

    def _record_events(self, events: List[Event]) -> None:
        for e in events:
            self.event_graph.add_event(e)
            self.working.add_event(e)
            if self.episode_builder is not None:
                self.episode_builder.on_event(e)
            for eid in e.entities:
                self.semantic.observe(eid, "event", e.type, weight=0.5)
        if self.plan is not None and self.episode_builder is not None:
            self.episode_builder.on_skill(self.plan.skill_name)

    def _update_spatial(self, state: WorldState) -> None:
        self.spatial_store.update(state)

    # ------------------------------------------------------------------ #
    # logging / checkpoints
    # ------------------------------------------------------------------ #
    def _log_line(self, step: int, total: int, report: PlayerReport, total_reward: float) -> str:
        loss = self.trainer.latest
        loss_str = " ".join(f"{k}={v:.4f}" for k, v in loss.items() if k in ("entity", "spatial", "total"))
        goal_str = f"{self.goal.goal_type}:{self.goal.status}" if self.goal else "-"
        return (
            f"[learn] {step}/{total} | loss {loss_str} | eps={self.eps:.3f} | "
            f"goal={goal_str} skill={self.plan.skill_name if self.plan else '-'} | "
            f"events={len(self.event_graph.nodes)} episodes={len(self.episodic)} "
            f"collected={self._collected_total} | reward_avg={total_reward / max(step, 1):.4f}"
        )

    def _summary_line(self, report: PlayerReport) -> str:
        return (
            f"[summary] steps={report.total_steps} wall={report.wall_time_s:.1f}s "
            f"events={report.events} episodes={report.episodes} goals_ok={report.goals_succeeded} "
            f"collected={report.collected_total} loss={report.final_loss.get('total', float('nan')):.4f} "
            f"mean_reward={report.mean_reward:+.4f}"
            + (f" checkpoint={report.checkpoint}" if report.checkpoint else "")
        )

    def save_checkpoint(self, path: str) -> str:
        return self.trainer.save_checkpoint(path, metrics=self.trainer.latest)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        return self.trainer.load_checkpoint(path)

    # convenience aliases used by the frozen API direction
    @property
    def memory(self) -> EpisodicMemory:
        return self.episodic

    @property
    def skills(self) -> SkillRegistry:
        return self.registry
