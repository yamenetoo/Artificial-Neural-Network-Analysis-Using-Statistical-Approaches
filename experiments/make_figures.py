import pandas as pd
from common import *
d = pd.read_csv(f"{RES}/e1_ci_coverage.csv")
lab = {"Bootstrap basic (paper Alg.7)": "Bootstrap basic (paper Alg. 7)", "CLT of replicates, B=30 (paper Alg.6)": "CLT of replicates, B=30 (paper Alg. 6)",
       "CLT of replicates, B=1000 (paper Alg.6)": "CLT of replicates, B=1000 (paper Alg. 6)"}
sty = ["-", "-", "-", "-", "--", ":", ":"]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, th in zip(axs, (0.80, 0.90, 0.95)):
    for i, (m, g) in enumerate(d[d.theta == th].groupby("method", sort=False)):
        ax.plot(g.n, g.coverage, marker="o", ms=3, ls=sty[i], label=lab.get(m, m), color=PAL[i % len(PAL)])
    ax.axhline(.95, color="k", lw=.8, ls="--"); ax.set_xscale("log"); ax.set_title(f"true accuracy $\\theta$ = {th}"); ax.set_xlabel("test-set size n")
axs[0].set_ylabel("empirical coverage (nominal 0.95)"); axs[0].set_ylim(0, 1.02)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, .13, 1, 1)); fig.savefig(f"{FIG}/e1_coverage.png")
df = pd.read_csv(f"{RES}/e4_reduction_summary.csv")
fig, axs = plt.subplots(1, 5, figsize=(12, 3.0), sharex=True)
order = ["lsc_qr", "lsc_medoid", "lsc_importance", "prune_only", "cluster_mean", "pca"]
nm = {"lsc_qr": "LSC-QR", "lsc_medoid": "LSC-medoid", "lsc_importance": "LSC-importance", "prune_only": "prune, no compensation", "cluster_mean": "cluster-mean (needs all k neurons)", "pca": "PCA (needs all k neurons)"}
for ax, ds in zip(axs, ["iris", "wine", "breast_cancer", "digits", "diabetes"]):
    g = df[df.dataset == ds]
    for i, mth in enumerate(order):
        gg = g[g.method == mth]
        ax.plot(gg.frac, gg["delta"] * 100, marker="o", ms=3, ls="--" if mth in ("cluster_mean", "pca") else "-", label=nm[mth], color=PAL[i])
    ax.axhline(0, color="k", lw=.7); ax.set_title(ds.replace("_", " ")); ax.set_xlabel("retained fraction"); ax.set_ylim(-30, 3)
axs[0].set_ylabel("change vs full net\n(accuracy pts; $R^2$ pts for diabetes)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(f"{FIG}/e4_reduction.png")
