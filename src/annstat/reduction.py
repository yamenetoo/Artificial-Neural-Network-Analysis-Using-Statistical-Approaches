"""Reducing the width of the last hidden layer.

Four strategies, all operating on activations ``H`` of a fitted network:

``cluster``   group neurons by hierarchical/k-means clustering and replace each group by an
              aggregate (paper Section 3.1.1); the output layer is re-fitted on the aggregates.
``pca``       replace the neurons by their leading principal components (paper Section 3.1.2).
``lsc``       **least-squares compensated pruning** (new): keep a subset S of neurons and absorb the
              contribution of every removed neuron into the kept ones through the regression
              ``h_P ~ B_0 + B_S h_S``; the new head is ``W_S + B_S W_P``. No retraining, the result
              is a genuine narrower network, and it is exact whenever the removed neurons are
              affine functions of the kept ones.
``select_*``  selectors for ``lsc``: rank-revealing pivoted QR (``qr``), ablation importance
              (``importance``) and cluster medoids (``medoid``).

Honest accounting matters: ``cluster``/``pca`` shrink the *head* but still evaluate all ``k`` neurons
to build their features, so ``n_neurons_needed`` stays ``k``; only ``lsc`` reduces the neurons that
must be computed at inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.linalg import qr
from scipy.spatial.distance import pdist

from ._utils import default_metric, softmax


@dataclass
class ReducedModel:
    name: str
    transform: Callable[[np.ndarray], np.ndarray]
    W: np.ndarray
    b: np.ndarray
    task: str
    classes_: Optional[np.ndarray]
    n_features: int
    n_neurons_needed: int
    info: dict = field(default_factory=dict)

    def logits(self, H):
        return self.transform(np.asarray(H, float)) @ self.W + self.b

    def predict_proba(self, H):
        return softmax(self.logits(H))

    def predict(self, H):
        z = self.logits(H)
        if self.task == "classification":
            return np.asarray(self.classes_)[np.argmax(z, 1)]
        return z.ravel() if z.shape[1] == 1 else z

    def n_head_params(self) -> int:
        return int(self.W.size + self.b.size)


# ---------------------------------------------------------------------------------------------
# head refitting
# ---------------------------------------------------------------------------------------------
def refit_head(Z, y, task: str, classes=None, C: float = 1e4, alpha: float = 1e-6):
    """Fit the output layer on features ``Z``: multinomial logistic regression or ridge."""
    if task == "classification":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=C, max_iter=10000)
        lr.fit(Z, y)
        if lr.coef_.shape[0] == 1:  # binary -> symmetrise
            W = np.hstack([-lr.coef_.T / 2, lr.coef_.T / 2])
            b = np.concatenate([-lr.intercept_ / 2, lr.intercept_ / 2])
        else:
            W, b = lr.coef_.T, lr.intercept_
        return W, b, lr.classes_
    from sklearn.linear_model import Ridge

    rg = Ridge(alpha=alpha).fit(Z, y)
    W = np.atleast_2d(rg.coef_).T if np.ndim(rg.coef_) == 1 else rg.coef_.T
    if W.ndim == 1:
        W = W[:, None]
    return W, np.atleast_1d(rg.intercept_), None


# ---------------------------------------------------------------------------------------------
# clustering and PCA
# ---------------------------------------------------------------------------------------------
def cluster_neurons(H, n_clusters: int, linkage: str = "ward", metric: str = "euclidean"):
    """Cluster neurons (columns of ``H``). Dead neurons get label ``-1``. Returns (labels, Z)."""
    H = np.asarray(H, float)
    alive = np.flatnonzero(H.std(0) > 1e-12)
    labels = -np.ones(H.shape[1], dtype=int)
    if len(alive) <= n_clusters:
        labels[alive] = np.arange(len(alive))
        return labels, None
    vecs = H[:, alive].T
    if linkage == "ward":
        metric = "euclidean"
    Z = hierarchy.linkage(pdist(vecs, metric=metric), method=linkage)
    lab = hierarchy.fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    labels[alive] = lab
    return labels, Z


def make_cluster_model(H, y, n_clusters: int, task: str, agg: str = "mean", linkage: str = "ward",
                       metric: str = "euclidean", classes=None) -> ReducedModel:
    labels, Z = cluster_neurons(H, n_clusters, linkage, metric)
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels[labels >= 0])]
    reducer = {"mean": lambda A: A.mean(1), "sum": lambda A: A.sum(1), "max": lambda A: A.max(1)}[agg]

    def transform(Hn):
        return np.column_stack([reducer(Hn[:, g]) for g in groups])

    W, b, cl = refit_head(transform(H), y, task, classes)
    return ReducedModel(f"cluster-{agg}", transform, W, b, task, cl if task == "classification" else None,
                        len(groups), H.shape[1], {"labels": labels, "linkage": Z})


def make_pca_model(H, y, n_components: int, task: str, classes=None) -> ReducedModel:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components).fit(H)

    def transform(Hn):
        return pca.transform(Hn)

    W, b, cl = refit_head(transform(H), y, task, classes)
    return ReducedModel("pca", transform, W, b, task, cl if task == "classification" else None,
                        n_components, H.shape[1],
                        {"explained_variance": float(pca.explained_variance_ratio_.sum())})


# ---------------------------------------------------------------------------------------------
# least-squares compensated pruning
# ---------------------------------------------------------------------------------------------
def select_qr(H, k: int) -> np.ndarray:
    """Pivoted-QR selection of ``k`` maximally independent (centred) neurons."""
    Hc = np.asarray(H, float) - np.mean(H, axis=0)
    _, _, piv = qr(Hc, pivoting=True, mode="economic")
    return np.sort(piv[:k])


def select_by_score(scores, k: int) -> np.ndarray:
    return np.sort(np.argsort(-np.asarray(scores))[:k])


def select_medoids(H, k: int, linkage: str = "ward") -> np.ndarray:
    """One representative (closest to the cluster mean) per neuron cluster."""
    H = np.asarray(H, float)
    labels, _ = cluster_neurons(H, k, linkage)
    keep = []
    for g in np.unique(labels[labels >= 0]):
        members = np.flatnonzero(labels == g)
        centre = H[:, members].mean(1, keepdims=True)
        keep.append(members[np.argmin(((H[:, members] - centre) ** 2).sum(0))])
    return np.sort(np.array(keep))


def compress_lsc(H_fit, W, b, keep, task: str = "classification", classes=None, ridge: float = 1e-8,
                 name: str = "lsc") -> ReducedModel:
    """Remove all neurons except ``keep`` and compensate with a least-squares regression."""
    H_fit = np.asarray(H_fit, float)
    keep = np.sort(np.asarray(keep, int))
    drop = np.setdiff1d(np.arange(H_fit.shape[1]), keep)
    Hs = H_fit[:, keep]
    if drop.size:
        A = np.column_stack([np.ones(len(Hs)), Hs])
        G = A.T @ A + ridge * np.eye(A.shape[1])
        Bm = np.linalg.solve(G, A.T @ H_fit[:, drop])  # (1+|S|, |P|)
        W_new = W[keep] + Bm[1:] @ W[drop]
        b_new = b + Bm[0] @ W[drop]
    else:
        W_new, b_new = W[keep], b
    return ReducedModel(name, lambda Hn, keep=keep: Hn[:, keep], W_new, b_new, task, classes,
                        len(keep), len(keep), {"keep": keep})


# ---------------------------------------------------------------------------------------------
def reduction_path(adapter, X_fit, y_fit, X_eval, y_eval, ks, methods=("lsc_qr", "lsc_importance", "lsc_medoid",
                   "cluster_mean", "pca"), importance_scores=None, incoming=None, seed=0) -> pd.DataFrame:
    """Evaluate reduction strategies over a grid of target sizes ``ks``.

    ``X_fit`` / ``y_fit`` are used to *construct* the reduction (selection, regression, head
    refit); ``X_eval`` / ``y_eval`` are held out and used only for scoring.
    """
    task = adapter.task
    metric = default_metric(task)
    H_fit = adapter.hidden(X_fit)
    H_ev = adapter.hidden(X_eval)
    W, b = adapter.head()
    k_full = H_fit.shape[1]
    base_pred = adapter.predict(X_eval)
    rows = [{"method": "full", "k": k_full, "k_requested": k_full, "metric": metric(y_eval, base_pred), "agreement": 1.0,
             "n_neurons_needed": k_full, "n_head_params": int(W.size + b.size),
             "n_hidden_layer_params_removed": 0}]
    if {"lsc_importance", "prune_only"} & set(methods) and importance_scores is None:
        from .neuron_tests import ablation_importance

        importance_scores = ablation_importance(adapter, X_fit, y_fit, B=200, rng=0)["loss_increase"].values
    for k in ks:
        if k >= k_full:
            continue
        for m in methods:
            if m == "lsc_qr":
                red = compress_lsc(H_fit, W, b, select_qr(H_fit, k), task, adapter.classes_, name=m)
            elif m == "lsc_importance":
                red = compress_lsc(H_fit, W, b, select_by_score(importance_scores, k), task, adapter.classes_, name=m)
            elif m == "lsc_medoid":
                red = compress_lsc(H_fit, W, b, select_medoids(H_fit, k), task, adapter.classes_, name=m)
            elif m == "cluster_mean":
                red = make_cluster_model(H_fit, y_fit, k, task, "mean", classes=adapter.classes_)
            elif m == "pca":
                red = make_pca_model(H_fit, y_fit, k, task, classes=adapter.classes_)
            elif m in ("thinet", "obs", "weight_norm", "lsc_leverage", "lsc_random", "merge"):
                from . import baselines as bl

                if m == "thinet":
                    red = bl.thinet_prune(H_fit, W, b, k, task, adapter.classes_)
                elif m == "obs":
                    red = bl.obs_prune(H_fit, W, b, k, task, adapter.classes_)[0]
                elif m == "weight_norm":
                    red = bl.weight_norm_prune(H_fit, W, b, k, task, adapter.classes_)
                elif m == "lsc_leverage":
                    red = compress_lsc(H_fit, W, b, bl.leverage_select(H_fit, k), task, adapter.classes_, name=m)
                elif m == "lsc_random":
                    kp = np.sort(np.random.default_rng(seed).choice(k_full, k, replace=False))
                    red = compress_lsc(H_fit, W, b, kp, task, adapter.classes_, name=m)
                else:
                    if incoming is None:
                        raise ValueError("method 'merge' needs incoming=(W_in, b_in)")
                    red = bl.neuron_merge(incoming[0], incoming[1], W, b, k, task, adapter.classes_)
            elif m == "prune_only":  # naive baseline: keep top-importance neurons, no compensation
                keep = np.sort(np.argsort(-np.asarray(importance_scores))[:k])
                red = ReducedModel(m, lambda Hn, keep=keep: Hn[:, keep], W[keep], b, task, adapter.classes_, k, k)
            else:
                raise ValueError(f"unknown method {m!r}")
            pred = red.predict(H_ev)
            rows.append({"method": m, "k": red.n_features, "k_requested": k, "metric": metric(y_eval, pred),
                         "agreement": float(np.mean(pred == base_pred)) if task == "classification" else np.nan,
                         "n_neurons_needed": red.n_neurons_needed, "n_head_params": red.n_head_params(),
                         "n_hidden_layer_params_removed": k_full - red.n_neurons_needed})
    return pd.DataFrame(rows)
