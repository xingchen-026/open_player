"""Small statistics helpers (mean/std, median, bootstrap CI, Welch t).

No new dependencies beyond scipy (already present).  Every headline
comparison in Phase 1.5 goes through is_improvement(); when the evidence is
weak the verdict is 'no reliable evidence', never hand-waved.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def mean_std(values: Sequence[float]) -> tuple:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    return (float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)


def median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(arr)) if len(arr) else float("nan")


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """Bootstrap 95% CI of the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def welch_t(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Welch's t-test (two-sided) between two samples."""
    from scipy import stats as sps
    t, p = sps.ttest_ind(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), equal_var=False)
    return {"t": float(t), "p": float(p)}


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Effect size (Cohen's d) between two samples."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    na, nb = len(arr_a), len(arr_b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * arr_a.var(ddof=1) + (nb - 1) * arr_b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((arr_a.mean() - arr_b.mean()) / pooled)


def is_improvement(a: Sequence[float], b: Sequence[float], alpha: float = 0.05, higher_is_better: bool = True, n_boot: int = 2000) -> Dict[str, Any]:
    """Compare a (candidate) vs b (baseline); verdict string, no packaging.

    Returns: mean_a, mean_b, std_a, std_b, median_a, median_b, delta_mean,
    cohens_d, p_value, ci95_a, verdict.
    verdict is one of:
      'improvement' / 'degradation' / 'no reliable evidence of improvement'
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    sign = 1.0 if higher_is_better else -1.0
    delta = (arr_a.mean() - arr_b.mean()) * sign
    ttest = welch_t(arr_a, arr_b)
    p = ttest["p"]
    ci = bootstrap_ci(arr_a, n_boot=n_boot, alpha=alpha)
    ci_lo, ci_hi = ci
    if delta > 0 and p < alpha:
        verdict = "improvement"
    elif delta < 0 and p < alpha:
        verdict = "degradation"
    else:
        verdict = "no reliable evidence of improvement"
    return {
        "mean_a": float(arr_a.mean()),
        "mean_b": float(arr_b.mean()),
        "std_a": float(arr_a.std(ddof=1)) if len(arr_a) > 1 else 0.0,
        "std_b": float(arr_b.std(ddof=1)) if len(arr_b) > 1 else 0.0,
        "median_a": float(np.median(arr_a)),
        "median_b": float(np.median(arr_b)),
        "delta_mean": float(delta),
        "cohens_d": cohens_d(arr_a, arr_b),
        "p_value": float(p),
        "ci95_a": [float(ci_lo), float(ci_hi)],
        "verdict": verdict,
    }
