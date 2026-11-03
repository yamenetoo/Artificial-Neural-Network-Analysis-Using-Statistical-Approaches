"""Interpretable statistical surrogates of the output layer.

The last layer of a network is already a (multinomial) logistic or linear model in the hidden
features.  Re-fitting that model by maximum likelihood on held-out activations gives what the
paper calls *logistic regression on hidden neurons*: coefficient estimates with Wald standard
errors, z-tests and confidence intervals, likelihood-ratio test of the whole layer, McFadden
pseudo-R^2, AIC and BIC.  Everything is implemented with NumPy/SciPy only.

``fidelity`` quantifies how faithfully the surrogate reproduces the network (agreement of
predictions) - a surrogate that is accurate *and* faithful can be read as an explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import encode_labels, softmax


@dataclass
class MultinomialLogit:
    coef: np.ndarray            # (p, q-1); reference class = classes_[0]
    cov: np.ndarray             # (p*(q-1), p*(q-1))
    classes_: np.ndarray
    feature_names: list
    loglik: float
    null_loglik: float
    n: int
    converged: bool
    n_iter: int
    ridge: float
    table: pd.DataFrame = field(repr=False, default=None)

    # ---- model-level statistics
    @property
    def n_params(self) -> int:
        return int(self.coef.size)

    @property
    def mcfadden_r2(self) -> float:
        return 1 - self.loglik / self.null_loglik

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.n_params

    @property
    def bic(self) -> float:
        return -2 * self.loglik + np.log(self.n) * self.n_params

    @property
    def lr_test(self) -> dict:
        stat = 2 * (self.loglik - self.null_loglik)
        df = self.coef.size - self.coef.shape[1]
        return {"statistic": float(stat), "df": int(df), "p_value": float(stats.chi2.sf(stat, df))}

    @property
    def separation_warning(self) -> bool:
        return bool(np.max(np.abs(self.coef)) > 20)

    def predict_proba(self, X) -> np.ndarray:
        Xd = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        z = np.column_stack([np.zeros(len(Xd)), Xd @ self.coef])
        return softmax(z)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def summary(self) -> dict:
        return {"n": self.n, "loglik": self.loglik, "mcfadden_r2": self.mcfadden_r2, "aic": self.aic,
                "bic": self.bic, "lr_test": self.lr_test, "converged": self.converged,
                "separation_warning": self.separation_warning}


def fit_multinomial_logit(X, y, classes=None, feature_names=None, ridge: float = 1e-6, max_iter: int = 100,
                          tol: float = 1e-9, level: float = 0.95) -> MultinomialLogit:
    """Newton-Raphson ML fit with a tiny ridge (stabilises quasi-separation) and Wald inference."""
    X = np.asarray(X, float)
    classes = np.unique(y) if classes is None else np.asarray(classes)
    yi = encode_labels(y, classes)
    n, k = X.shape
    q = len(classes)
    m = q - 1
    p = k + 1
    Xd = np.column_stack([np.ones(n), X])
    Y = np.eye(q)[yi]
    names = ["const"] + (list(feature_names) if feature_names is not None else [f"h{j}" for j in range(k)])

    def loglik_pen(B):
        z = np.column_stack([np.zeros(n), Xd @ B])
        zmax = z.max(1, keepdims=True)
        lse = zmax[:, 0] + np.log(np.exp(z - zmax).sum(1))
        ll = (Y * z).sum() - lse.sum()
        return ll - 0.5 * ridge * (B[1:] ** 2).sum(), ll

    B = np.zeros((p, m))
    cur, _ = loglik_pen(B)
    converged = False
    it = 0
    pen = ridge * np.eye(p)
    pen[0, 0] = 0.0  # do not penalise the intercept
    for it in range(1, max_iter + 1):
        P = softmax(np.column_stack([np.zeros(n), Xd @ B]))
        grad = Xd.T @ (Y[:, 1:] - P[:, 1:]) - ridge * np.vstack([np.zeros((1, m)), B[1:]])
        Hb = np.zeros((m * p, m * p))
        for a in range(m):
            for c in range(m):
                w = P[:, a + 1] * ((a == c) - P[:, c + 1])
                Hb[a * p:(a + 1) * p, c * p:(c + 1) * p] = -(Xd * w[:, None]).T @ Xd - (pen if a == c else 0)
        g = grad.T.ravel()
        try:
            delta = np.linalg.solve(-Hb, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(-Hb, g, rcond=None)[0]
        step, improved = 1.0, False
        for _ in range(30):
            Bn = B + step * delta.reshape(m, p).T
            new, _ = loglik_pen(Bn)
            if new >= cur - 1e-12:
                improved = True
                break
            step /= 2
        if not improved:
            break
        B, prev, cur = Bn, cur, new
        if abs(cur - prev) < tol * (1 + abs(cur)):
            converged = True
            break

    # information matrix at the solution
    P = softmax(np.column_stack([np.zeros(n), Xd @ B]))
    Hb = np.zeros((m * p, m * p))
    for a in range(m):
        for c in range(m):
            w = P[:, a + 1] * ((a == c) - P[:, c + 1])
            Hb[a * p:(a + 1) * p, c * p:(c + 1) * p] = -(Xd * w[:, None]).T @ Xd - (pen if a == c else 0)
    cov = np.linalg.pinv(-Hb)
    _, ll = loglik_pen(B)
    freq = Y.mean(0).clip(1e-12)
    ll0 = float((Y * np.log(freq)).sum())

    se = np.sqrt(np.clip(np.diag(cov), 0, None)).reshape(m, p).T
    z = B / np.where(se > 0, se, np.nan)
    pv = 2 * stats.norm.sf(np.abs(z))
    crit = stats.norm.ppf(1 - (1 - level) / 2)
    rows = []
    for a in range(m):
        for j in range(p):
            rows.append({"class": classes[a + 1], "term": names[j], "coef": B[j, a], "se": se[j, a], "z": z[j, a],
                         "p_value": pv[j, a], "ci_low": B[j, a] - crit * se[j, a], "ci_high": B[j, a] + crit * se[j, a]})
    return MultinomialLogit(B, cov, classes, names, float(ll), ll0, n, converged, it, ridge, pd.DataFrame(rows))


def fit_ols(X, y, feature_names=None, robust: bool = True, level: float = 0.95) -> dict:
    """OLS of a single response on hidden features with (HC3) robust standard errors."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ y
    res = y - Xd @ beta
    if robust:
        h = np.einsum("ij,jk,ik->i", Xd, XtX_inv, Xd)
        meat = (Xd * (res / (1 - h))[:, None]).T @ (Xd * (res / (1 - h))[:, None])
        cov = XtX_inv @ meat @ XtX_inv
    else:
        cov = XtX_inv * (res @ res) / (n - k - 1)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    df = n - k - 1
    names = ["const"] + (list(feature_names) if feature_names is not None else [f"h{j}" for j in range(k)])
    crit = stats.t.ppf(1 - (1 - level) / 2, df)
    table = pd.DataFrame({"term": names, "coef": beta, "se": se, "t": t, "p_value": 2 * stats.t.sf(np.abs(t), df),
                          "ci_low": beta - crit * se, "ci_high": beta + crit * se})
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    return {"table": table, "r2": float(r2), "n": n, "robust": robust}


def fidelity(surrogate_pred, network_pred, y=None) -> dict:
    """Agreement between surrogate and network predictions (and optional accuracies)."""
    sp, npd = np.asarray(surrogate_pred), np.asarray(network_pred)
    out = {"agreement": float(np.mean(sp == npd))}
    if y is not None:
        out["surrogate_accuracy"] = float(np.mean(sp == np.asarray(y)))
        out["network_accuracy"] = float(np.mean(npd == np.asarray(y)))
    return out


def drop_one_lr_tests(X, y, classes=None, ridge: float = 1e-6, alpha: float = 0.05) -> pd.DataFrame:
    """Likelihood-ratio test for each hidden neuron: refit the multinomial logit without it.

    Wald z-tests are unreliable under (quasi-)separation and strong collinearity - exactly the
    regime of a well-trained network evaluated on its training data (Hauck-Donner effect). The
    drop-one likelihood-ratio test does not rely on the curvature at the (possibly infinite) MLE.
    P-values are Benjamini-Hochberg adjusted.
    """
    from .multitest import adjust

    X = np.asarray(X, float)
    full = fit_multinomial_logit(X, y, classes, ridge=ridge)
    df_ = len(full.classes_) - 1
    rows = []
    for j in range(X.shape[1]):
        keep = [i for i in range(X.shape[1]) if i != j]
        red = fit_multinomial_logit(X[:, keep], y, classes, ridge=ridge)
        stat = max(2 * (full.loglik - red.loglik), 0.0)
        rows.append({"neuron": j, "lr_statistic": stat, "df": df_, "p_value": float(stats.chi2.sf(stat, df_))})
    out = pd.DataFrame(rows)
    out["p_adj"] = adjust(out["p_value"].values, "bh")
    out["significant"] = out["p_adj"] < alpha
    return out
