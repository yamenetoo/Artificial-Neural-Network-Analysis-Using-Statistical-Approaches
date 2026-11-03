import json, numpy as np, pandas as pd
from common import *
import annstat as A
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
out = {}
for name, loader, w, test_size in (("iris", load_iris, 10, .2), ("breast_cancer", load_breast_cancer, 16, .3)):
    d = loader(); X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = MLPClassifier(hidden_layer_sizes=(w,), max_iter=3000, random_state=0).fit(Xtr, ytr)
    a = A.SklearnMLPAdapter(m)
    rep = A.StatisticalTuner(m, seed=0).analyze(Xtr, ytr, Xte, yte, n_boot=1000, n_perm=200, reduction_ks=[2, 3, 4, 6, 8],
            feature_names=list(d.feature_names), class_names=list(map(str, a.classes_)), min_effect=0.01)
    open(f"{RES}/e5_{name}_report.md", "w").write(rep.to_markdown())
    H, (W, b) = a.hidden(Xte), a.head()
    r = {"n_test": len(yte), "width": w, "accuracy": rep.performance["wilson"].estimate,
         "wilson": [rep.performance["wilson"].lower, rep.performance["wilson"].upper],
         "boot": [rep.performance["bootstrap-percentile"].lower, rep.performance["bootstrap-percentile"].upper],
         "effective_rank": rep.redundancy["effective_rank"], "n_corr_pairs": rep.redundancy["n_highly_correlated_pairs"],
         "sep_sig": int(rep.class_separation.significant.sum()), "abl_sig": int(rep.ablation.significant.sum()),
         "abl_table": rep.ablation.round(4).to_dict("records")}
    if rep.contribution is not None:
        ind = A.contribution_tests(H, W, paired=False, correction="holm")["pairwise"]
        pr = rep.contribution["pairwise"]
        r.update(n_pairs=len(pr), mwu_raw=int((ind.p_value < .05).sum()), mwu_holm=int(ind.significant.sum()),
                 wil_raw=int((pr.p_value < .05).sum()), wil_holm=int(pr.significant.sum()))
    s = rep.surrogate.summary(); r["surrogate"] = {k: v for k, v in s.items()}
    r["surrogate_fidelity"] = rep.surrogate_fidelity
    tab = rep.surrogate.table
    r["surrogate_sig_coef_raw"] = int((tab[tab.term != "const"].p_value < .05).sum())
    r["surrogate_n_coef"] = int((tab.term != "const").sum())
    lr = A.drop_one_lr_tests(a.hidden(Xtr), ytr, a.classes_)
    r["drop_one_lr_sig"] = int(lr.significant.sum()); r["drop_one_lr_n"] = len(lr)
    r["inputs"] = rep.inputs.round(4).to_dict("records")
    r["reduction"] = rep.reduction.round(4).to_dict("records")
    out[name] = r
    if name == "iris":
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.1))
        A.plots.hidden_correlation_heatmap(H, axs[0]); axs[0].set_title("(a) hidden-neuron correlations")
        ab = rep.ablation; cols = [PAL[0] if s_ else "#BBBBBB" for s_ in ab.significant]
        axs[1].bar(ab.neuron.astype(str), ab.loss_increase, color=cols)
        axs[1].errorbar(ab.neuron.astype(str), ab.loss_increase, yerr=[ab.loss_increase - ab.ci_low, ab.ci_high - ab.loss_increase], fmt="none", ecolor="k", lw=.7)
        axs[1].set_xlabel("hidden neuron"); axs[1].set_ylabel("log-loss increase when ablated"); axs[1].set_title("(b) ablation importance")
        rd = rep.reduction
        for i, (mm, g) in enumerate(rd[rd.method != "full"].groupby("method")):
            axs[2].plot(g.k, g.metric, marker="o", ms=3, label=mm, color=PAL[i])
        axs[2].axhline(rd[rd.method == "full"].metric.iloc[0], color="k", ls="--", lw=.8, label="full network")
        axs[2].set_xlabel("retained features k"); axs[2].set_ylabel("test accuracy"); axs[2].set_title("(c) width reduction"); axs[2].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"{FIG}/e5_iris.png")
json.dump(out, open(f"{RES}/e5_case_studies.json", "w"), indent=1, default=float)
for k, v in out.items():
    print(k, {x: v[x] for x in v if x not in ("abl_table", "inputs", "reduction")})
# E4 figure
df = pd.read_csv(f"{RES}/e4_reduction_summary.csv")
fig, axs = plt.subplots(1, 5, figsize=(12, 2.8), sharex=True)
order = ["lsc_qr", "lsc_medoid", "lsc_importance", "prune_only", "cluster_mean", "pca"]
lab = {"lsc_qr": "LSC-QR", "lsc_medoid": "LSC-medoid", "lsc_importance": "LSC-importance", "prune_only": "prune, no compensation", "cluster_mean": "cluster-mean*", "pca": "PCA*"}
for ax, ds in zip(axs, ["iris", "wine", "breast_cancer", "digits", "diabetes"]):
    g = df[df.dataset == ds]
    for i, mth in enumerate(order):
        gg = g[g.method == mth]
        ax.plot(gg.frac, gg["delta"], marker="o", ms=3, ls="--" if mth in ("cluster_mean", "pca") else "-", label=lab[mth], color=PAL[i])
    ax.axhline(0, color="k", lw=.7); ax.set_title(ds.replace("_", " ")); ax.set_xlabel("retained fraction")
axs[0].set_ylabel("change in test metric vs full net")
axs[-1].legend(fontsize=6, loc="lower right"); fig.tight_layout(); fig.savefig(f"{FIG}/e4_reduction.png")
