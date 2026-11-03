"""Multiple-testing corrections.

The original notebooks run dozens of hypothesis tests (neurons x output pairs) at a raw
alpha = 0.05.  Every inference routine in ``annstat`` therefore adjusts its p-values by default.
"""
from __future__ import annotations

import numpy as np


def bonferroni(p) -> np.ndarray:
    p = np.asarray(p, float)
    return np.minimum(p * p.size, 1.0)


def holm(p) -> np.ndarray:
    """Holm step-down adjustment (controls the family-wise error rate)."""
    p = np.asarray(p, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def benjamini_hochberg(p) -> np.ndarray:
    """Benjamini-Hochberg step-up adjustment (controls the false discovery rate)."""
    p = np.asarray(p, float)
    m = p.size
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for rank, i in enumerate(order):
        k = m - rank
        running = min(running, p[i] * m / k)
        adj[i] = running
    return np.minimum(adj, 1.0)


def benjamini_yekutieli(p) -> np.ndarray:
    p = np.asarray(p, float)
    c = np.sum(1.0 / np.arange(1, p.size + 1))
    return np.minimum(benjamini_hochberg(p) * c, 1.0)


_METHODS = {
    "none": lambda p: np.asarray(p, float),
    "bonferroni": bonferroni,
    "holm": holm,
    "bh": benjamini_hochberg,
    "fdr_bh": benjamini_hochberg,
    "by": benjamini_yekutieli,
}


def adjust(p, method: str = "holm") -> np.ndarray:
    try:
        return _METHODS[method](p)
    except KeyError as exc:
        raise ValueError(f"unknown correction {method!r}; choose from {sorted(_METHODS)}") from exc
