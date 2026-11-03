"""Significance of input variables via a permutation test of the trained network.

Null hypothesis for feature *j*: the trained network does not use ``x_j`` - permuting that column
(breaking its relation to the response and the other features) does not increase the expected loss.
The p-value is ``(1 + #{permuted loss <= base loss}) / (R + 1)``; adjustment is Benjamini-Hochberg.

This is a model-reliance test for the *fitted* network, not a test of the population relation
(for that see Horel & Giesecke, 2020). With strongly correlated inputs, permutation creates
unrealistic samples - interpret accordingly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import as_rng, per_sample_loss
from .multitest import adjust


def _loss(adapter, X, y):
    if adapter.task == "classification":
        return per_sample_loss(adapter.predict_proba(X), y, "classification", adapter.classes_).mean()
    return per_sample_loss(adapter.predict(X), y, "regression").mean()


def permutation_input_test(adapter, X, y, n_perm: int = 200, alpha: float = 0.05,
                           feature_names=None, rng=None) -> pd.DataFrame:
    rng = as_rng(rng)
    X = np.asarray(X, float)
    n, d = X.shape
    names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(d)]
    base = _loss(adapter, X, y)
    rows = []
    for j in range(d):
        losses = np.empty(n_perm)
        Xp = X.copy()
        for r in range(n_perm):
            Xp[:, j] = X[rng.permutation(n), j]
            losses[r] = _loss(adapter, Xp, y)
        rows.append({
            "feature": names[j],
            "loss_increase": float(losses.mean() - base),
            "p_value": float((1 + np.sum(losses <= base)) / (n_perm + 1)),
        })
    df = pd.DataFrame(rows)
    df["p_adj"] = adjust(df["p_value"].values, "bh")
    df["significant"] = df["p_adj"] < alpha
    df.attrs["base_loss"] = float(base)
    return df
