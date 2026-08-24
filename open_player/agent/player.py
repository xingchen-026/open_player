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
from open_player.events.detector import HeuristicEventDetector, HybridEventDetector
from open_player.events.graph import EventGraph
from open_player.memory.episodic import EpisodicMemory, EpisodeBuilder
from open_player.memory.procedural import ProceduralMemory
from open_player.memory.semantic import SemanticMemory
from open_player.memory.spatial import SpatialMemoryStore
from open_player.memory.working import WorkingMemory
from open_player.motivation.goals import GoalManager
from open_player.motivation.intrinsic import IntrinsicReward, VisitCounter
from open_player.motivation.motivation import IntrinsicMotivation
from open_player.observation.dummy import DummyVisionEncoder
from open_player.observation.vision import LearnedVisionEncoder
from open_player.planning.planner import Plan, Planner
from open_player.skills.neural import NeuralSkill, StateFeaturizer
from open_player.skills.registry import SkillRegistry
from open_player.tracking.tracker import BeliefTracker
from open_player.training.skill_trainer import SkillTrainer, SkillTrainReport
from open_player.training.trainer import WorldModelTrainer
from open_player.world.model import WorldModel

log = logging.getLogger("open_player.agent")


def _compact(d: Any) -> Any:
    """Recursively round floats for readable JSON output."""
    if isinstance(d, dict):
        return {k: _compact(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_compact(v) for v in d]
    if isinstance(d, float):
        return round(d, 4)
    return d


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

        # Phase 1: learned vision (opt-in via config.vision.enabled)
        self.vision_enabled = bool(config.get("vision.enabled", False))
        self.vision: Optional[LearnedVisionEncoder] = None
        if self.vision_enabled:
            self.vision = LearnedVisionEncoder(self.schema, config, device=self.device).to(self.device)

        # Phase 1: intrinsic reward + visit counter
        self.intrinsic = IntrinsicReward(config) if "intrinsic" in config else None
        self.visit_counter = VisitCounter(cap=int(config.get("intrinsic.visit_count_cap", 10))) if self.intrinsic is not None else None

        # action / skill wiring (env-specific controller attached in attach())
        self.controller = ActionController(DiscreteActionSpace(list(DEFAULT_ACTION_NAMES)))
        self.registry = SkillRegistry.build_default(self.controller, config, intrinsic=self.intrinsic, visit_counter=self.visit_counter)

        # world model + trainer
        self.world_model = WorldModel(self.schema, config, num_actions=len(self.controller.space.names))
        self.world_model.to(self.device)
        self.trainer = WorldModelTrainer(self.world_model, config, self.schema, device=self.device)

        # cognition
        if bool(config.get("event_pred.enabled", False)):
            self.detector = HybridEventDetector(
                HeuristicEventDetector(), world_model=self.world_model,
                conf_blend=float(config.get("event_pred.conf_blend", 0.5)), device=self.device,
            )
        else:
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

        # Phase 1: skill training machinery + learned skill slot
        self.featurizer = StateFeaturizer(self.schema, device=self.device)
        self.neural_skill: Optional[NeuralSkill] = None

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
        self._prev_uncertainty: Optional[float] = None
        self._prev_action: Optional[int] = None

    # ------------------------------------------------------------------ #
    # wiring
    # ------------------------------------------------------------------ #
    def attach(self, environment: Environment) -> None:
        """Bind an environment; rebuild the action-dependent parts."""
        self.env = environment
        names = environment.action_space.names
        self.controller = ActionController(DiscreteActionSpace(list(names)))
        self.registry = SkillRegistry.build_default(self.controller, self.config, intrinsic=self.intrinsic, visit_counter=self.visit_counter)
        if self.neural_skill is not None:
            self.registry.register(self.neural_skill)
        if self.world_model.num_actions != len(names):
            self.world_model = WorldModel(self.schema, self.config, num_actions=len(names))
            self.world_model.to(self.device)
            self.trainer = WorldModelTrainer(self.world_model, self.config, self.schema, device=self.device)
            if self.detector is not None and isinstance(self.detector, HybridEventDetector):
                self.detector.world_model = self.world_model
        self.planner = Planner(self.config, self.registry, self.schema, model=self.world_model, procedural_memory=self.procedural)

    def reset_cognition(self) -> None:
        """Clear per-run cognitive state (episode/goal caches)."""
        self.working.clear()
        self.motivation.reset()
        self.spatial_store.reset()
        if self.visit_counter is not None:
            self.visit_counter.reset()
        self._episode_counter = 0
        self._goal_counter = 0
        self._collected_total = 0
        self._last_error = None
        self._goal_report = None
        self._prev_uncertainty = None
        self._prev_action = None
        self._cum_intrinsic = 0.0
        self.eps = float(self.config.get("training.exploration_eps", 0.15))

    # ------------------------------------------------------------------ #
    # perception / decision
    # ------------------------------------------------------------------ #
    def perceive(self, observation: Observation, t: int = 0) -> WorldState:
        """Observation -> WorldState.

        Phase 0: BeliefTracker (cross-step belief update).
        Phase 1 (vision.enabled): LearnedVisionEncoder (RGB -> learned
        spatial/entity features, fresh per step).
        """
        if self.vision_enabled and self.vision is not None:
            self.state = self.vision.encode(observation, t=t)
        else:
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
            # (the hybrid detector needs the action to query the predictor)
            info["action"] = action.index
            events = self.detector.detect(state, state2, info, step + 1)
            change = 1.0 if any(e.type in SEMANTIC_EVENT_TYPES for e in events) else 0.0
            self._record_events(events)

            # world model learning (1-step + multi-step + learned change)
            metrics = self.trainer.online_step(state, action.index, state2, float(reward), bool(done), change)
            if metrics:
                self._last_error = float(metrics.get("entity", self._last_error or 0.0))
            tick = self.trainer.tick()
            if self.vision_enabled:
                # each vision-built state's graph is consumed by its update
                state = state.detach()
                state2 = state2.detach()

            # Phase 1: intrinsic reward + visit counts
            env_info = dict(info)
            env_info["action"] = action.index
            env_info["world_model_error"] = self._last_error or 0.0
            env_info["num_resources"] = getattr(getattr(environment, "world", None), "num_resources", 1)
            intrinsic_reward = 0.0
            if self.intrinsic is not None:
                player = next((e for e in state2.entity_states(0) if e.semantic_type == "player"), None)
                player_pos = None if player is None else player.position
                if self.visit_counter is not None and player_pos is not None:
                    self.visit_counter.update(player_pos)
                ir = self.intrinsic.compute(
                    state=state2,
                    world_model_error=self._last_error or 0.0,
                    uncertainty_mean=self.trainer.uncertainty.mean,
                    prev_uncertainty_mean=self._prev_uncertainty,
                    action=action.index,
                    prev_action=self._prev_action,
                    visit_counter=self.visit_counter,
                    player_pos=player_pos,
                )
                intrinsic_reward = float(ir["total"])
                env_info["intrinsic_reward"] = intrinsic_reward
                env_info["intrinsic_novelty"] = float(ir["novelty"])
                env_info["uncertainty_mean"] = float(self.trainer.uncertainty.mean)
                self._prev_uncertainty = float(self.trainer.uncertainty.mean)
                self._prev_action = action.index
                self._cum_intrinsic = self._cum_intrinsic + intrinsic_reward if hasattr(self, "_cum_intrinsic") else intrinsic_reward

            # motivation / goal bookkeeping
            self.working.last_reward = float(reward)
            self.working.add_state(state2)
            self.working.add_action(action.index)
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
        if self.vision is not None:
            report.extra["vision_params"] = self.vision.num_parameters()
        if self.neural_skill is not None:
            report.extra["neural_skill_params"] = self.neural_skill.num_parameters()
        if hasattr(self, "_cum_intrinsic"):
            report.extra["mean_intrinsic_reward"] = self._cum_intrinsic / max(total_steps, 1)
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
            info["action"] = action.index
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

    # ------------------------------------------------------------------ #
    # Phase 1 public APIs (additive, backward compatible)
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        environment: Environment,
        episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate the current policy (no weight updates) for N episodes."""
        from open_player.evaluation.metrics import episode_metrics, summarize_metrics
        self.attach(environment)
        self.trainer.eval()
        episodes = episodes or int(self.config.get("evaluation.eval_episodes", 4))
        max_steps = max_steps or int(self.config.get("evaluation.eval_max_steps", 100))
        entries: List[Dict[str, Any]] = []
        for ep in range(episodes):
            obs = environment.reset(seed=seed + 101 * ep)
            state = self.perceive(obs, 0)
            self.goal = None
            self.plan = None
            total_reward = 0.0
            env_info: Dict[str, Any] = {"threat_level": 0.0}
            info: Dict[str, Any] = {}
            succeeded_any = False
            for t in range(max_steps):
                action = self.choose_action(state, env_info, explore=False)
                obs2, reward, done, info = environment.step(action)
                total_reward += float(reward)
                state = self.perceive(obs2, t + 1)
                env_info = dict(info)
                env_info["num_resources"] = getattr(getattr(environment, "world", None), "num_resources", 1)
                if self.goal is not None:
                    status = self.goal_manager.update(self.goal, state, env_info)
                    if status == "succeeded":
                        succeeded_any = True
                if done:
                    break
            entry = episode_metrics(environment, info, total_reward, t + 1)
            entry["goal_type"] = self.goal.goal_type if self.goal else None
            entry["goal_status"] = self.goal.status if self.goal else None
            entry["goal_succeeded"] = bool(succeeded_any or (self.goal is not None and self.goal.status == "succeeded"))
            entries.append(entry)
        summary = summarize_metrics(entries)
        summary["agent"] = "phase1" if self.vision_enabled else "phase0"
        summary["episodes"] = episodes
        return summary

    def train_skill(
        self,
        environment: Optional[Environment] = None,
        steps: Optional[int] = None,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> SkillTrainReport:
        """Behavior cloning: rule trajectories -> NeuralSkill, then register it."""
        from open_player.evaluation.baselines import RuleBaseline
        env = environment or self.env
        if env is None:
            raise RuntimeError("train_skill needs an environment")
        self.attach(env)
        steps = steps or int(self.config.get("skill_training.train_steps", 400))
        sc = self.config.get("skill_training", {})
        skill_name = str(sc.get("skill_name", "neural_explore"))
        trainer = SkillTrainer(self.config, self.featurizer, device=self.device)
        policy = RuleBaseline(env.action_space, self.schema, seed=int(self.config.seed))
        xs, actions, terms, info = trainer.collect(env, self.perceive, policy.act, steps, seed=int(self.config.seed))
        skill = NeuralSkill(
            name=skill_name,
            action_names=list(env.action_space.names),
            featurizer=self.featurizer,
            horizon=int(self.config.get("planning.horizons.medium", 8)),
        ).to(self.device)
        report = trainer.train(skill, xs, actions, terms)
        report.kept_episodes = int(info["kept_episodes"])
        report.data_steps = int(info["data_steps"])
        if save_path:
            report.save_path = trainer.save(skill, save_path)
        self.neural_skill = skill
        self.registry.register(skill)
        if verbose:
            print(f"[train_skill] {report.to_dict()}", flush=True)
        return report

    def evaluate_transfer(
        self,
        train_env: Environment,
        test_env: Environment,
        steps: Optional[int] = None,
        adaptation_steps: Optional[int] = None,
        episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train on World A, zero-shot eval World B, short adaptation, re-eval."""
        import json, os
        from open_player.evaluation.benchmark import evaluate_baseline, evaluate_world_model
        steps = steps or int(self.config.get("training.steps", 2000))
        adaptation_steps = adaptation_steps if adaptation_steps is not None else int(self.config.get("evaluation.adaptation_steps", 1000))
        episodes = episodes or int(self.config.get("evaluation.eval_episodes", 4))
        max_steps = max_steps or int(self.config.get("evaluation.eval_max_steps", 100))

        # baselines on World B (independent of the agent)
        results: Dict[str, Any] = {"world_b": {}, "world_a": {}}
        results["baselines_b"] = {
            "random": evaluate_baseline(test_env, "random", self.perceive, episodes, max_steps, self.schema, seed=int(self.config.seed)),
            "rule": evaluate_baseline(test_env, "rule", self.perceive, episodes, max_steps, self.schema, seed=int(self.config.seed)),
        }
        results["baselines_b"]["phase0_agent"] = self._phase0_eval(test_env, episodes, max_steps)

        # train on World A
        rep = self.learn(train_env, total_steps=steps, verbose=False)
        results["training"] = {"steps": steps, "final_loss": dict(rep.final_loss), "collected": rep.collected_total}
        results["world_a"] = self.evaluate(train_env, episodes=episodes, max_steps=max_steps)
        # world model prediction errors on A and B
        results["prediction_errors_a"] = evaluate_world_model(self.world_model, self.perceive, train_env)
        results["prediction_errors_b_zero_shot"] = evaluate_world_model(self.world_model, self.perceive, test_env)

        # zero-shot on World B
        results["world_b"]["zero_shot"] = self.evaluate(test_env, episodes=episodes, max_steps=max_steps)

        # short adaptation on World B
        self.learn(test_env, total_steps=adaptation_steps, verbose=False)
        results["world_b"]["after_adaptation"] = self.evaluate(test_env, episodes=episodes, max_steps=max_steps)
        results["prediction_errors_b_adapted"] = evaluate_world_model(self.world_model, self.perceive, test_env)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "transfer_results.json"), "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, default=float)
        if verbose:
            print(json.dumps(_compact(results), indent=2), flush=True)
        return results

    def _phase0_eval(self, env: Environment, episodes: int, max_steps: int) -> Dict[str, Any]:
        """Phase 0-style agent evaluation (structured path, no vision)."""
        from open_player.core.config import Config as _Cfg
        base = self.config.to_dict()
        base.setdefault("vision", {})["enabled"] = False
        from open_player.agent.player import Player as _Player
        p0 = _Player(_Cfg(base))
        return p0.evaluate(env, episodes=episodes, max_steps=max_steps, seed=int(self.config.seed))

    def save_checkpoint(self, path: str) -> str:
        modules: Dict[str, Any] = {}
        if self.vision is not None:
            modules["vision"] = self.vision
        if self.world_model.change_predictor is not None:
            modules["change_predictor"] = self.world_model.change_predictor
        if self.neural_skill is not None:
            modules["neural_skill"] = self.neural_skill
        return self.trainer.save_checkpoint(path, metrics=self.trainer.latest, extra_modules=modules)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        modules: Dict[str, Any] = {}
        if self.vision is not None:
            modules["vision"] = self.vision
        if self.world_model.change_predictor is not None:
            modules["change_predictor"] = self.world_model.change_predictor
        if self.neural_skill is not None:
            modules["neural_skill"] = self.neural_skill
        return self.trainer.load_checkpoint(path, modules=modules)

    # convenience aliases used by the frozen API direction
    @property
    def memory(self) -> EpisodicMemory:
        return self.episodic

    @property
    def skills(self) -> SkillRegistry:
        return self.registry
