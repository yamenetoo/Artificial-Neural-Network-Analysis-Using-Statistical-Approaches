"""Confidence intervals for the performance of a trained network.

Four families are provided:

* closed-form binomial intervals (Wilson, Clopper-Pearson) for the accuracy of a *fixed* model,
* bootstrap of the evaluation set (no retraining) - interval for the fixed model's accuracy,
* bootstrap with retraining and out-of-bag evaluation - includes training variability,
* the two procedures of the original paper, kept for reproducibility:
  ``clt_ci`` (Algorithm 6) and ``bootstrap_interval(..., method="basic")`` (Algorithm 7).

.. warning::
   ``clt_ci`` applied to *bootstrap replicates* returns ``mean +/- t * s / sqrt(B)``. Its width
   shrinks to zero as the number of replicates ``B`` grows, so it is an interval for the *mean of
   the replicates*, not for the model's generalisation accuracy. Use it only as a baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import stats

from ._utils import as_rng, default_metric, stratified_indices


@dataclass
class CI:
    estimate: float
    lower: float
    upper: float
    level: float
    method: str
    extra: dict = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CI({self.method}: {self.estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}] @ {self.level:.0%})"


def binomial_ci(successes: int, n: int, level: float = 0.95, method: str = "wilson") -> CI:
    """Interval for a proportion (accuracy of a fixed classifier on ``n`` i.i.d. test cases)."""
    a = 1 - level
    p = successes / n
    if method == "wilson":
        z = stats.norm.ppf(1 - a / 2)
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        lo, hi = centre - half, centre + half
    elif method in ("clopper-pearson", "exact"):
        lo = 0.0 if successes == 0 else stats.beta.ppf(a / 2, successes, n - successes + 1)
        hi = 1.0 if successes == n else stats.beta.ppf(1 - a / 2, successes + 1, n - successes)
    elif method == "wald":
        z = stats.norm.ppf(1 - a / 2)
        half = z * np.sqrt(p * (1 - p) / n)
        lo, hi = p - half, p + half
    else:
        raise ValueError("method must be 'wilson', 'clopper-pearson' or 'wald'")
    return CI(p, float(max(lo, 0.0)), float(min(hi, 1.0)), level, f"binomial-{method}", {"n": n})


def clt_ci(values, level: float = 0.95, dist: str = "t") -> CI:
    """Paper Algorithm 6: ``mean +/- crit * s / sqrt(len(values))`` (see module warning)."""
    v = np.asarray(values, float)
    n = v.size
    m, s = v.mean(), v.std(ddof=1)
    a = 1 - level
    crit = stats.t.ppf(1 - a / 2, n - 1) if dist == "t" else stats.norm.ppf(1 - a / 2)
    half = crit * s / np.sqrt(n)
    return CI(float(m), float(m - half), float(m + half), level, f"clt-{dist}", {"sd": float(s), "n": n})


def bootstrap_interval(values, estimate: Optional[float] = None, level: float = 0.95,
                       method: str = "percentile") -> CI:
    """Interval from bootstrap replicates.

    ``percentile``: empirical quantiles of the replicates.
    ``basic``     : reverse percentile ``2*theta_hat - q_{1-a/2}, 2*theta_hat - q_{a/2}`` (paper Alg. 7).
    ``normal``    : ``theta_hat +/- z * sd(replicates)``.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    est = float(v.mean()) if estimate is None else float(estimate)
    a = 1 - level
    q_lo, q_hi = np.quantile(v, [a / 2, 1 - a / 2])
    if method == "percentile":
        lo, hi = q_lo, q_hi
    elif method == "basic":
        lo, hi = 2 * est - q_hi, 2 * est - q_lo
    elif method == "normal":
        z = stats.norm.ppf(1 - a / 2)
        sd = v.std(ddof=1)
        lo, hi = est - z * sd, est + z * sd
    else:
        raise ValueError("method must be 'percentile', 'basic' or 'normal'")
    return CI(est, float(lo), float(hi), level, f"bootstrap-{method}", {"B": int(v.size)})


def bootstrap_metric(
    adapter,
    X,
    y,
    scheme: str = "resample_eval",
    B: int = 500,
    level: float = 0.95,
    interval: str = "percentile",
    metric: Optional[Callable] = None,
    stratify: bool = True,
    warm_start: bool = True,
    max_iter: Optional[int] = None,
    rng=None,
) -> CI:
    """Bootstrap interval for a performance metric (accuracy / R^2 by default).

    scheme
        ``resample_eval``  resample (X, y) pairs and score the *fixed* model - cheap, interval for
                           the accuracy of this model.
        ``oob_retrain``    for each replicate retrain (warm start) on a bootstrap sample, score on
                           the out-of-bag cases - includes training variability.
        ``paper_retrain``  as above but score on the bootstrap sample itself (paper Alg. 7); this
                           is optimistic because the model is evaluated on its own training data.
    """
    rng = as_rng(rng)
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    metric = metric or default_metric(adapter.task)
    use_strata = stratify and adapter.task == "classification"
    reps = []

    if scheme == "resample_eval":
        pred = adapter.predict(X)
        for _ in range(B):
            idx = stratified_indices(y, rng) if use_strata else rng.integers(0, n, n)
            reps.append(metric(y[idx], pred[idx]))
        est = metric(y, pred)
    elif scheme in ("oob_retrain", "paper_retrain"):
        est = metric(y, adapter.predict(X))
        for _ in range(B):
            idx = stratified_indices(y, rng) if use_strata else rng.integers(0, n, n)
            model_b = adapter.refit(X[idx], y[idx], warm_start=warm_start, max_iter=max_iter, rng=rng)
            if scheme == "paper_retrain":
                reps.append(metric(y[idx], model_b.predict(X[idx])))
            else:
                oob = np.setdiff1d(np.arange(n), np.unique(idx))
                if oob.size:
                    reps.append(metric(y[oob], model_b.predict(X[oob])))
    else:
        raise ValueError("scheme must be 'resample_eval', 'oob_retrain' or 'paper_retrain'")

    ci = bootstrap_interval(reps, estimate=est, level=level, method=interval)
    ci.method = f"{scheme}/{interval}"
    ci.extra["replicates"] = np.asarray(reps)
    return ci


def accuracy_intervals(adapter, X, y, level: float = 0.95, B: int = 1000, rng=None) -> dict:
    """Convenience: the recommended set of intervals for a fixed classifier."""
    pred = adapter.predict(X)
    k = int(np.sum(np.asarray(y) == pred))
    out = {
        "wilson": binomial_ci(k, len(y), level, "wilson"),
        "clopper-pearson": binomial_ci(k, len(y), level, "clopper-pearson"),
        "bootstrap-percentile": bootstrap_metric(adapter, X, y, "resample_eval", B, level, "percentile", rng=rng),
    }
    return out
