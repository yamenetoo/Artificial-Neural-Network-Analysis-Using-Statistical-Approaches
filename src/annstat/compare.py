"""Statistical comparison of methods over several data sets / models (Demsar 2006; Benavoli et al. 2017).

``scores`` is a DataFrame with one row per *block* (data set or model) and one column per method.
``friedman_test``      Friedman test with the Iman-Davenport F correction and mean ranks.
``nemenyi_cd``         Nemenyi critical difference of mean ranks.
``pairwise_wilcoxon``  Wilcoxon signed-rank tests of a control method against all others (or all
                       pairs), Holm/BH-adjusted, with median paired difference.
``paired_seed_tests``  Within-block paired tests over repetitions (seeds / splits).
Ranks are computed so that rank 1 = best; set ``higher_is_better=False`` for losses.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .multitest import adjust


def _ranks(scores: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    s = scores if not higher_is_better else -scores
    return s.rank(axis=1, method="average")


def friedman_test(scores: pd.DataFrame, higher_is_better: bool = True) -> dict:
    """Friedman chi-square and the less conservative Iman-Davenport F statistic."""
    N, k = scores.shape
    if N < 2 or k < 3:
        raise ValueError("need at least 2 blocks and 3 methods")
    R = _ranks(scores, higher_is_better)
    mean_rank = R.mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(mean_rank**2) - k * (k + 1) ** 2 / 4)
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) > chi2 else np.inf
    return {"n_blocks": N, "n_methods": k, "chi2": float(chi2), "p_chi2": float(stats.chi2.sf(chi2, k - 1)),
            "F_iman_davenport": float(ff), "p_F": float(stats.f.sf(ff, k - 1, (k - 1) * (N - 1))),
            "mean_rank": mean_rank.sort_values()}


def nemenyi_cd(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Critical difference of mean ranks for the Nemenyi post-hoc test."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def _wilcoxon(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    if np.allclose(d, 0):
        return np.nan, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st, p = stats.wilcoxon(d, zero_method="wilcox", method="auto")
    return float(st), float(p)


def pairwise_wilcoxon(scores: pd.DataFrame, control: str | None = None, correction: str = "holm",
                      alpha: float = 0.05) -> pd.DataFrame:
    """Paired Wilcoxon tests; ``control`` vs every other method, or all pairs if ``control`` is None."""
    cols = list(scores.columns)
    pairs = [(control, c) for c in cols if c != control] if control else list(itertools.combinations(cols, 2))
    rows = []
    for a, b in pairs:
        d = scores[a] - scores[b]
        st, p = _wilcoxon(scores[a], scores[b])
        rows.append({"a": a, "b": b, "median_diff": float(d.median()), "mean_diff": float(d.mean()),
                     "wins_a": int((d > 0).sum()), "wins_b": int((d < 0).sum()), "statistic": st, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_adj"] = adjust(out["p_value"].values, correction)
    out["significant"] = out["p_adj"] < alpha
    return out


def paired_seed_tests(long: pd.DataFrame, block: str, method: str, repeat: str, metric: str, control: str,
                      correction: str = "holm", alpha: float = 0.05) -> pd.DataFrame:
    """Within each block, test ``control`` against every other method over the repetitions (paired Wilcoxon).
    P-values are adjusted within each block."""
    out = []
    for b, g in long.groupby(block):
        wide = g.pivot(index=repeat, columns=method, values=metric)
        t = pairwise_wilcoxon(wide, control=control, correction=correction, alpha=alpha)
        t.insert(0, block, b)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def compare_methods(scores: pd.DataFrame, control: str | None = None, higher_is_better: bool = True,
                    alpha: float = 0.05, correction: str = "holm") -> dict:
    """Friedman + Nemenyi CD + Holm-adjusted Wilcoxon of ``control`` (default: best mean rank) vs the rest."""
    fr = friedman_test(scores, higher_is_better)
    control = control or fr["mean_rank"].index[0]
    return {"friedman": fr, "cd": nemenyi_cd(scores.shape[1], scores.shape[0], alpha), "control": control,
            "pairwise": pairwise_wilcoxon(scores, control, correction, alpha)}
