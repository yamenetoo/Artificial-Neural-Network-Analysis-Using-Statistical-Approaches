"""E2: retraining-bootstrap schemes - own-sample evaluation (paper Alg.7) is optimistic."""
import numpy as np, pandas as pd
from common import *
import annstat as A
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
rows = []
for name, loader, w in (("iris", load_iris, 10), ("wine", load_wine, 16), ("breast_cancer", load_breast_cancer, 16)):
    X, y = loader(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = MLPClassifier(hidden_layer_sizes=(w,), max_iter=2000, random_state=0).fit(Xtr, ytr)
    a = A.SklearnMLPAdapter(m)
    rng = np.random.default_rng(1)
    paper, oob, cold = [], [], []
    for b in range(150):
        from annstat._utils import stratified_indices
        idx = stratified_indices(ytr, rng)
        mb = a.refit(Xtr[idx], ytr[idx], warm_start=True, max_iter=100, rng=rng)
        paper.append(np.mean(mb.predict(Xtr[idx]) == ytr[idx]))
        o = np.setdiff1d(np.arange(len(ytr)), np.unique(idx))
        oob.append(np.mean(mb.predict(Xtr[o]) == ytr[o]))
        if b < 60:  # cold start: no information flows from the full-data model
            mc = a.refit(Xtr[idx], ytr[idx], warm_start=False, rng=rng)
            cold.append(np.mean(mc.predict(Xtr[o]) == ytr[o]))
    hold = np.mean(a.predict(Xte) == yte)
    w_ci = A.binomial_ci(int(np.sum(a.predict(Xte) == yte)), len(yte))
    for nm, v in (("own-sample, warm start (paper Alg.7)", paper), ("out-of-bag, warm start", oob), ("out-of-bag, cold start", cold)):
        ci = A.bootstrap_interval(v, method="percentile")
        rows.append({"dataset": name, "scheme": nm, "mean": np.mean(v), "ci_low": ci.lower, "ci_high": ci.upper})
    rows.append({"dataset": name, "scheme": "hold-out accuracy (Wilson)", "mean": hold, "ci_low": w_ci.lower, "ci_high": w_ci.upper})
df = pd.DataFrame(rows); df.to_csv(f"{RES}/e2_bootstrap_schemes.csv", index=False); print(df.round(4))
