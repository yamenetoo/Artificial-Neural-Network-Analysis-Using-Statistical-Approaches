"""E8: logit-reconstruction error of LSC: in-sample bound (Prop. 4) and out-of-sample behaviour."""
import numpy as np, pandas as pd
from common import *
import annstat as A
from scipy.linalg import qr
from sklearn.datasets import load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
rows = []
for name, loader, w in (("wine", load_wine, 16), ("breast_cancer", load_breast_cancer, 16), ("digits", load_digits, 32)):
    X, y = loader(return_X_y=True)
    for seed in range(10):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=seed, stratify=y)
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = MLPClassifier((w,), max_iter=1500, random_state=seed).fit(Xtr, ytr); a = A.SklearnMLPAdapter(m)
        Htr, Hte = a.hidden(Xtr), a.hidden(Xte); W, b = a.head()
        Ztr, Zte = Htr @ W + b, Hte @ W + b
        for frac in (.25, .5, .75):
            mm = max(2, round(w * frac))
            keep = A.select_qr(Htr, mm); drop = np.setdiff1d(np.arange(w), keep)
            red = A.compress_lsc(Htr, W, b, keep)
            tr = ((Ztr - red.logits(Htr)) ** 2).sum(1).mean(); te = ((Zte - red.logits(Hte)) ** 2).sum(1).mean()
            # in-sample bound: ||R||_F^2 ||W_P||_2^2 / n, R from pivoted QR of the centred activations
            Hc = Htr - Htr.mean(0); _, Rq, piv = qr(Hc, pivoting=True, mode="economic")
            R22 = Rq[mm:, mm:]
            bound = np.linalg.norm(R22, "fro") ** 2 * np.linalg.norm(W[drop], 2) ** 2 / len(Htr)
            sv = np.linalg.svd(Hc, compute_uv=False)
            opt_tail = (sv[mm:] ** 2).sum() / len(Htr)   # best rank-m residual energy (Eckart-Young)
            rows.append({"dataset": name, "seed": seed, "frac": frac, "m": mm, "train_mse": tr, "test_mse": te, "bound": bound,
                         "qr_resid": np.linalg.norm(R22, "fro") ** 2 / len(Htr), "opt_tail": opt_tail,
                         "logit_var": Ztr.var(0).sum()})
pd.DataFrame(rows).to_csv(f"{RES}/e8_bound.csv", index=False)
