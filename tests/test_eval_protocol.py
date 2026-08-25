"""Eval protocol / metrics collector / repro helpers."""
from __future__ import annotations

import json
import os

from open_player.evaluation.protocol import EvalProtocol, MetricsCollector, aggregate_rows
from open_player.evaluation.repro import config_fingerprint, git_hash, make_run_dir, new_experiment_id, save_csv, save_json
from open_player.evaluation.stats import bootstrap_ci, cohens_d, is_improvement, mean_std, median, welch_t


def test_protocol_defaults():
    p = EvalProtocol(seeds=[0, 1, 2], episodes=3)
    assert len(p.seed_episodes()) == 9
    d = p.to_dict()
    assert d["max_steps"] == 100


def test_aggregate_rows_mean_std_median():
    rows = [{"a": 1.0, "b": 10.0}, {"a": 3.0, "b": 20.0}, {"a": 5.0, "b": 30.0}]
    s = aggregate_rows(rows)
    assert s["n"] == 3
    assert abs(s["a_mean"] - 3.0) < 1e-9
    assert abs(s["a_median"] - 3.0) < 1e-9
    assert abs(s["b_mean"] - 20.0) < 1e-9
    assert s["a_std"] > 0


def test_metrics_collector():
    mc = MetricsCollector()
    mc.add({"x": 1})
    mc.add({"x": 2})
    assert len(mc) == 2
    assert abs(mc.summary()["x_mean"] - 1.5) < 1e-9


def test_stats_helpers():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    m, s = mean_std(a)
    assert m == 3.0 and s > 0
    assert median(a) == 3.0
    lo, hi = bootstrap_ci(a, n_boot=500)
    assert lo < m < hi
    t = welch_t(a, [5.0, 6.0, 7.0, 8.0, 9.0])
    assert t["p"] < 0.05
    assert cohens_d([5.0, 6.0, 7.0], [1.0, 2.0, 3.0]) > 1.0


def test_is_improvement_verdicts():
    base = [1.0, 1.1, 0.9, 1.0, 1.2]
    better = [1.8, 1.9, 1.7, 2.0, 1.6]
    r = is_improvement(better, base, higher_is_better=True)
    assert r["verdict"] == "improvement"
    r2 = is_improvement(base, base, higher_is_better=True)
    assert r2["verdict"] == "no reliable evidence of improvement"
    r3 = is_improvement(base, better, higher_is_better=True)
    assert r3["verdict"] == "degradation"


def test_repro_helpers(tmp_path):
    gh = git_hash(8)
    assert len(gh) == 8
    fp1 = config_fingerprint({"a": 1})
    fp2 = config_fingerprint({"a": 2})
    assert fp1 != fp2 and len(fp1) == 8
    exp_id = new_experiment_id(config={"a": 1})
    assert exp_id.startswith("p1.5-")
    run_dir = make_run_dir(str(tmp_path / "results"), config={"a": 1}, exp_id=exp_id)
    assert os.path.exists(os.path.join(run_dir, "run_info.json"))
    with open(os.path.join(run_dir, "run_info.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    assert info["exp_id"] == exp_id
    p1 = save_json(run_dir, "metrics.json", {"x": 1})
    p2 = save_csv(run_dir, "metrics.csv", [{"x": 1, "y": 2}])
    assert os.path.exists(p1) and os.path.exists(p2)
