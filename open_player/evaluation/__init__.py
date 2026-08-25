"""Phase 1 evaluation framework + Phase 1.5 scientific validation tools."""
from __future__ import annotations

from open_player.evaluation.baselines import RandomBaseline, RuleBaseline
from open_player.evaluation.benchmark import evaluate_agent, evaluate_baseline, evaluate_world_model
from open_player.evaluation.logger import ExperimentLogger
from open_player.evaluation.metrics import episode_metrics, summarize_metrics
from open_player.evaluation.protocol import EvalProtocol, MetricsCollector, aggregate_rows
from open_player.evaluation.repro import git_hash, make_run_dir, new_experiment_id, save_csv, save_json
from open_player.evaluation.stats import bootstrap_ci, cohens_d, is_improvement, mean_std, median, welch_t
from open_player.evaluation.world_model_baselines import PersistenceWorldModel, RandomDynamicsWorldModel

__all__ = [
    "EvalProtocol",
    "ExperimentLogger",
    "MetricsCollector",
    "PersistenceWorldModel",
    "RandomBaseline",
    "RandomDynamicsWorldModel",
    "RuleBaseline",
    "aggregate_rows",
    "bootstrap_ci",
    "cohens_d",
    "episode_metrics",
    "evaluate_agent",
    "evaluate_baseline",
    "evaluate_world_model",
    "git_hash",
    "is_improvement",
    "make_run_dir",
    "mean_std",
    "median",
    "new_experiment_id",
    "save_csv",
    "save_json",
    "summarize_metrics",
    "welch_t",
]
