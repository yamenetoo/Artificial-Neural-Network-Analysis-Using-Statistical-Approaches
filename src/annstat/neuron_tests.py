"""Inference about individual hidden neurons.

Generalises the two notebooks of the original repository:

* ``contribution_tests``      - does neuron *k* push the output logits differently?  (notebook:
  Kruskal-Wallis and pairwise Mann-Whitney on ``h_k * w_kj``.)  Pairs of output columns are
  computed on the *same* samples, so a paired test (Wilcoxon / Friedman) is the statistically
  appropriate default; the independent-sample tests are kept via ``paired=False``.
* ``class_separation_tests``  - does the activation of neuron *k* differ between classes (or
  correlate with the response for regression)?  Kruskal-Wallis / Spearman with an effect size.
* ``ablation_importance``     - how much does the loss increase when neuron *k* is switched off?
  Paired bootstrap confidence interval and one-sided p-value.
* ``redundancy_summary``      - correlation structure and effective rank of the hidden layer.

All families of p-values are adjusted (Holm / Benjamini-Hochberg) - the raw 0.05 threshold of the
notebooks is not family-wise valid for ``k * q(q-1)/2`` simultaneous tests.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import as_rng, per_sample_loss
from .multitest import adjust


def weighted_outputs(H: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Array ``(k, n, q)`` with entries ``h_{i,k} * w_{k,j}`` (the notebooks' ``AllOutput``)."""
    return np.einsum("nk,kq->knq", H, W)


# ----------------------------------------------------------------------------------------------
def contribution_tests(H, W, paired: bool = True, correction: str = "holm", alpha: float = 0.05,
                       class_names=None) -> dict:
    """Omnibus + pairwise tests of neuron contributions to the output logits.

    Returns ``{"omnibus": DataFrame, "pairwise": DataFrame}``. Needs at least 3 output columns for
    the omnibus test; binary problems (symmetrised head) are better served by
    :func:`class_separation_tests`.
    """
    H = np.asarray(H, float)
    W = np.asarray(W, float)
    k, q = W.shape
    n = H.shape[0]
    names = list(class_names) if class_names is not None else [f"O{j+1}" for j in range(q)]
    C = weighted_outputs(H, W)
    omni, pair = [], []
    for i in range(k):
        cols = [C[i, :, j] for j in range(q)]
        stat = p = np.nan
        if q >= 3 and np.ptp(C[i]) > 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p = stats.friedmanchisquare(*cols) if paired else stats.kruskal(*cols)
            except ValueError:
                stat, p = np.nan, 1.0
        omni.append({"neuron": i, "test": "friedman" if paired else "kruskal", "statistic": stat, "p_value": p})
        for a, c in itertools.combinations(range(q), 2):
            x, z = cols[a], cols[c]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if paired:
                        if np.allclose(x, z):
                            raise ValueError
                        st, pv = stats.wilcoxon(x, z)
                        d = x - z
                        d = d[d != 0]
                        ranks = stats.rankdata(np.abs(d))
                        eff = (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()
                    else:
                        st, pv = stats.mannwhitneyu(x, z, alternative="two-sided")
                        eff = 2 * st / (n * n) - 1
            except ValueError:
                st, pv, eff = np.nan, 1.0, 0.0
            pair.append({"neuron": i, "pair": f"{names[a]} vs {names[c]}",
                         "test": "wilcoxon" if paired else "mannwhitney",
                         "statistic": st, "p_value": pv, "effect_rank_biserial": eff})
    omni = pd.DataFrame(omni)
    pair = pd.DataFrame(pair)
    ok = omni["p_value"].notna()
    omni["p_adj"] = np.nan
    if ok.any():
        omni.loc[ok, "p_adj"] = adjust(omni.loc[ok, "p_value"].values, "bh" if correction in ("bh", "fdr_bh") else correction)
    pair["p_adj"] = adjust(pair["p_value"].values, correction)
    omni["significant"] = omni["p_adj"] < alpha
    pair["significant"] = pair["p_adj"] < alpha
    return {"omnibus": omni, "pairwise": pair}


# ----------------------------------------------------------------------------------------------
def class_separation_tests(H, y, task: str = "classification", alpha: float = 0.05) -> pd.DataFrame:
    """Per-neuron test of association with the response, BH-adjusted.

    classification: Kruskal-Wallis across classes, effect size epsilon^2 = H / (n - 1).
    regression:     Spearman correlation, effect size |rho|.
    """
    H = np.asarray(H, float)
    y = np.asarray(y)
    n, k = H.shape
    rows = []
    for i in range(k):
        h = H[:, i]
        if np.ptp(h) == 0:  # dead neuron on this sample
            rows.append({"neuron": i, "statistic": 0.0, "p_value": 1.0, "effect": 0.0, "dead": True})
            continue
        if task == "classification":
            groups = [h[y == c] for c in np.unique(y)]
            st, p = stats.kruskal(*groups)
            eff = st / (n - 1)
        else:
            st, p = stats.spearmanr(h, np.asarray(y, float).ravel())
            eff = abs(st)
        rows.append({"neuron": i, "statistic": float(st), "p_value": float(p), "effect": float(eff), "dead": False})
    df = pd.DataFrame(rows)
    df["p_adj"] = adjust(df["p_value"].values, "bh")
    df["significant"] = df["p_adj"] < alpha
    df.attrs["effect_name"] = "epsilon_squared" if task == "classification" else "abs_spearman"
    return df


# ----------------------------------------------------------------------------------------------
def ablation_importance(adapter, X, y, mode: str = "zero", B: int = 2000, level: float = 0.95,
                        alpha: float = 0.05, min_effect: float = 0.0, rng=None) -> pd.DataFrame:
    """Loss increase when a single hidden neuron is removed (``zero`` or replaced by its ``mean``).

    The per-sample loss differences ``d_i = loss_i(ablated) - loss_i(full)`` are paired, hence a
    paired bootstrap gives the confidence interval; the one-sided p-value is the bootstrap
    probability that the mean increase is not positive. P-values are BH-adjusted.
    ``min_effect`` adds a *practical* significance requirement: a neuron is flagged significant only
    if the adjusted p-value is below ``alpha`` **and** the mean loss increase is at least ``min_effect``
    (statistical significance alone flags tiny but reliable effects as important).
    """
    rng = as_rng(rng)
    H = adapter.hidden(X)
    W, b = adapter.head()
    classes = adapter.classes_
    base_out = adapter.output_from_hidden(H, W, b)
    base_loss = per_sample_loss(base_out, y, adapter.task, classes)
    base_pred = np.argmax(base_out, 1) if adapter.task == "classification" else None
    yi = None
    if adapter.task == "classification":
        yi = np.searchsorted(np.asarray(classes), np.asarray(y))
    n, k = H.shape
    idx = rng.integers(0, n, size=(B, n))
    rows = []
    for j in range(k):
        Hj = H.copy()
        Hj[:, j] = 0.0 if mode == "zero" else H[:, j].mean()
        out = adapter.output_from_hidden(Hj, W, b)
        d = per_sample_loss(out, y, adapter.task, classes) - base_loss
        boots = d[idx].mean(axis=1)
        a = 1 - level
        lo, hi = np.quantile(boots, [a / 2, 1 - a / 2])
        p = (1 + np.sum(boots <= 0)) / (B + 1)
        row = {"neuron": j, "loss_increase": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p)}
        if adapter.task == "classification":
            acc_full = float(np.mean(base_pred == yi))
            acc_abl = float(np.mean(np.argmax(out, 1) == yi))
            row["accuracy_drop"] = acc_full - acc_abl
        rows.append(row)
    df = pd.DataFrame(rows)
    df["p_adj"] = adjust(df["p_value"].values, "bh")
    df["significant"] = (df["p_adj"] < alpha) & (df["loss_increase"] >= min_effect)
    return df


# ----------------------------------------------------------------------------------------------
def redundancy_summary(H, corr_threshold: float = 0.95) -> dict:
    """Correlation structure of a hidden layer.

    ``effective_rank`` is exp(entropy of the normalised eigenvalues of the activation covariance);
    ``participation_ratio`` is (sum lambda)^2 / sum lambda^2. Both lie in [1, k]; values far below
    ``k`` indicate a redundant layer.
    """
    H = np.asarray(H, float)
    alive = H.std(axis=0) > 1e-12
    Ha = H[:, alive]
    k = H.shape[1]
    out = {"n_neurons": k, "n_dead": int((~alive).sum())}
    if Ha.shape[1] < 2:
        out.update(effective_rank=float(Ha.shape[1]), participation_ratio=float(Ha.shape[1]),
                   n_highly_correlated_pairs=0, corr=np.eye(Ha.shape[1]))
        return out
    corr = np.corrcoef(Ha, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(np.cov(Ha, rowvar=False)), 0, None)
    p = lam / lam.sum()
    p = p[p > 0]
    iu = np.triu_indices_from(corr, 1)
    out.update(
        effective_rank=float(np.exp(-(p * np.log(p)).sum())),
        participation_ratio=float(lam.sum() ** 2 / (lam**2).sum()),
        n_highly_correlated_pairs=int(np.sum(np.abs(corr[iu]) >= corr_threshold)),
        corr=corr,
    )
    return out
