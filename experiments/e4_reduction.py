"""E4: width reduction of a trained one-hidden-layer network, 5 datasets x 15 seeds."""
import numpy as np, pandas as pd, time
from common import *
import annstat as A
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
cfg = [("iris", load_iris, 12, "c"), ("wine", load_wine, 16, "c"), ("breast_cancer", load_breast_cancer, 16, "c"),
       ("digits", load_digits, 32, "c"), ("diabetes", load_diabetes, 16, "r")]
methods = ("lsc_qr", "lsc_importance", "lsc_medoid", "prune_only", "cluster_mean", "pca")
rows = []
for name, loader, w, kind in cfg:
    X, y = loader(return_X_y=True)
    for seed in range(15):
        t0 = time.time()
        st = y if kind == "c" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=seed, stratify=st)
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        if kind == "c":
            m = MLPClassifier(hidden_layer_sizes=(w,), max_iter=1500, random_state=seed).fit(Xtr, ytr)
        else:
            m = MLPRegressor(hidden_layer_sizes=(w,), max_iter=3000, random_state=seed, alpha=1e-2).fit(Xtr, ytr)
        a = A.SklearnMLPAdapter(m)
        ks = sorted({max(2, round(w * f)) for f in (.25, .5, .75)})
        df = A.reduction_path(a, Xtr, ytr, Xte, yte, ks, methods)
        df["dataset"], df["seed"], df["width"] = name, seed, w
        rows.append(df)
    print(name, "done", round(time.time() - t0, 1), "s/seed", flush=True)
out = pd.concat(rows); out.to_csv(f"{RES}/e4_reduction_raw.csv", index=False)
full = out[out.method == "full"].set_index(["dataset", "seed"])["metric"]
o = out[out.method != "full"].copy()
o["frac"] = (o.k_requested / o.width).round(2)
o["delta"] = o.apply(lambda r: r.metric - full[(r.dataset, r.seed)], axis=1)
agg = o.groupby(["dataset", "frac", "method"]).agg(metric=("metric", "mean"), delta=("delta", "mean"), delta_sd=("delta", "std"),
        agreement=("agreement", "mean"), neurons_needed=("n_neurons_needed", "mean")).reset_index()
agg.to_csv(f"{RES}/e4_reduction_summary.csv", index=False)
print(agg.round(3).to_string())
