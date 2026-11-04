"""Reference compression baselines operating on a last hidden layer (activations ``H``, head ``W, b``).

All return a :class:`~annstat.reduction.ReducedModel` (or a keep-set) so they plug into
``reduction_path``.  They are faithful *reimplementations of the published ideas on a single
hidden layer*, not ports of the original code bases:

``select_thinet``     greedy removal that minimises the energy of the removed logit contributions
                      (ThiNet, Luo et al. 2017) followed by a per-neuron least-squares *rescaling*.
``obs_prune``         sequential layer-wise Optimal Brain Surgeon (Hassibi & Stork 1993; Dong et al.
                      2017): remove the neuron with smallest saliency, update all remaining weights.
``weight_norm_prune`` data-free magnitude baseline, no compensation.
``leverage_select``   CUR-style deterministic leverage-score selection (Mahoney & Drineas 2009).
``neuron_merge``      data-free merging by weight similarity (Srinivas & Babu 2015; Kim et al. 2020).
"""
from __future__ import annotations

import numpy as np

from .reduction import ReducedModel, compress_lsc


def _rm(name, keep, W, b, task, classes):
    keep = np.sort(np.asarray(keep, int))
    return ReducedModel(name, lambda Hn, keep=keep: Hn[:, keep], W, b, task, classes, len(keep), len(keep),
                        {"keep": keep})


def weight_norm_prune(H, W, b, m, task="classification", classes=None):
    keep = np.sort(np.argsort(-np.linalg.norm(W, axis=1))[:m])
    return _rm("weight_norm", keep, W[keep], b, task, classes)


def select_thinet(H, W, m):
    """ThiNet greedy selection: repeatedly remove the neuron whose addition to the removed set
    gives the smallest energy ``||sum_removed h_j W_j||_F^2``. Returns (keep, scales)."""
    H = np.asarray(H, float)
    n, k = H.shape
    C = np.einsum("nk,kq->knq", H, W)          # (k, n, q) contributions to the logits
    Cf = C.reshape(k, -1)
    sq = (Cf ** 2).sum(1)
    removed, acc = [], np.zeros(Cf.shape[1])
    remaining = list(range(k))
    while len(remaining) > m:
        r = np.array(remaining)
        energy = 2.0 * (Cf[r] @ acc) + sq[r]    # ||acc + C_j||^2 - ||acc||^2, vectorised over candidates
        best = int(r[int(np.argmin(energy))])
        removed.append(best); remaining.remove(best); acc = acc + Cf[best]
    keep = np.sort(np.array(remaining))
    Ck = C[keep].reshape(len(keep), -1)
    Z0 = C.sum(0).reshape(-1)
    G = Ck @ Ck.T + 1e-10 * np.eye(len(keep))
    s = np.linalg.solve(G, Ck @ Z0)            # per-neuron rescaling (ThiNet's final step)
    return keep, s


def thinet_prune(H, W, b, m, task="classification", classes=None):
    keep, s = select_thinet(H, W, m)
    return _rm("thinet", keep, W[keep] * s[:, None], b, task, classes)


def obs_prune(H, W, b, m, task="classification", classes=None, ridge=1e-8):
    """Sequential layer-wise OBS on the head. Returns the compensated head and the keep-set."""
    H = np.asarray(H, float)
    n, k = H.shape
    A = np.column_stack([np.ones(n), H])
    Hinv = np.linalg.inv(A.T @ A / n + ridge * np.eye(k + 1))
    Wf = np.vstack([b[None, :], W]).astype(float)         # (k+1, q) incl. bias row 0
    alive = list(range(1, k + 1))
    while len(alive) > m:
        sal = [(float(Wf[j] @ Wf[j]) / (2 * Hinv[j, j]), j) for j in alive]
        _, j = min(sal)
        Wf = Wf - np.outer(Hinv[:, j], Wf[j]) / Hinv[j, j]
        Wf[j] = 0.0
        Hinv = Hinv - np.outer(Hinv[:, j], Hinv[j, :]) / Hinv[j, j]
        alive.remove(j)
    keep = np.array(sorted(a - 1 for a in alive))
    return _rm("obs", keep, Wf[keep + 1], Wf[0], task, classes), keep


def leverage_select(H, m, rank=None):
    """Deterministic CUR-style selection: top-``m`` rank-``r`` column leverage scores (r = m)."""
    Hc = np.asarray(H, float) - np.mean(H, axis=0)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    r = min(rank or m, Vt.shape[0])
    lev = (Vt[:r] ** 2).sum(0)
    return np.sort(np.argsort(-lev)[:m])


def neuron_merge(W_in, b_in, W, b, m, task="classification", classes=None):
    """Data-free merging: keep the ``m`` neurons with largest incoming-weight norm and fold every pruned
    neuron into its most cosine-similar kept neuron with the least-squares scale of the incoming
    weights (positive-homogeneous activations assumed, e.g. ReLU). ``W_in`` is ``(d, k)``."""
    V = np.vstack([W_in, np.asarray(b_in)[None, :]]).T       # (k, d+1) incoming weights per neuron
    norms = np.linalg.norm(V, axis=1)
    keep = np.sort(np.argsort(-norms)[:m])
    drop = np.setdiff1d(np.arange(len(V)), keep)
    Wn = W[keep].copy()
    Vk = V[keep]
    for p in drop:
        cos = Vk @ V[p] / (np.linalg.norm(Vk, axis=1) * norms[p] + 1e-12)
        i = int(np.argmax(cos))
        s = max(float(Vk[i] @ V[p] / (Vk[i] @ Vk[i] + 1e-12)), 0.0)
        Wn[i] += s * W[p]
    return _rm("merge", keep, Wn, b, task, classes)
