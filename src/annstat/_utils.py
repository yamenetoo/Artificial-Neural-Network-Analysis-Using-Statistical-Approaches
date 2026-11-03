"""Small numerical helpers shared by all modules."""
from __future__ import annotations

import numpy as np


def as_rng(seed=None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def encode_labels(y, classes) -> np.ndarray:
    """Map labels to integer indices of ``classes`` (sorted unique labels)."""
    classes = np.asarray(classes)
    lookup = {c: i for i, c in enumerate(classes.tolist())}
    return np.array([lookup[v] for v in np.asarray(y).tolist()], dtype=int)


def per_sample_loss(proba_or_pred: np.ndarray, y, task: str, classes=None) -> np.ndarray:
    """Per-sample loss: negative log-likelihood (classification) or squared error."""
    if task == "classification":
        yi = encode_labels(y, classes)
        p = np.clip(proba_or_pred[np.arange(len(yi)), yi], 1e-12, 1.0)
        return -np.log(p)
    y = np.asarray(y, dtype=float).reshape(len(y), -1)
    pr = np.asarray(proba_or_pred, dtype=float).reshape(len(y), -1)
    return ((y - pr) ** 2).sum(axis=1)


def default_metric(task: str):
    """accuracy for classification, R^2 for regression."""
    if task == "classification":
        return lambda yt, yp: float(np.mean(np.asarray(yt) == np.asarray(yp)))

    def r2(yt, yp):
        yt = np.asarray(yt, float).ravel()
        yp = np.asarray(yp, float).ravel()
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return r2


def stratified_indices(y, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap indices drawn with replacement *within* each class."""
    y = np.asarray(y)
    idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        idx.append(rng.choice(members, size=len(members), replace=True))
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out
