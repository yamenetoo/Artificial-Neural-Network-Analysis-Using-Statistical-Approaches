"""E1: coverage / width of accuracy intervals for a FIXED classifier.
Correct/incorrect indicators of a fixed model on i.i.d. test data are Bernoulli(theta), so the
number of hits is Binomial(n, theta) - the simulation is exact for any network."""
import numpy as np, pandas as pd
from common import *
import annstat as A
from scipy import stats
rng = np.random.default_rng(2026)
R, B = 5000, 1000
rows = []
for theta in (0.80, 0.90, 0.95, 0.99):
    for n in (30, 100, 300, 1000):
        hits = {k: 0 for k in ["Wald", "Wilson", "Clopper-Pearson", "Bootstrap percentile", "Bootstrap basic (paper Alg.7)", "CLT of replicates, B=30 (paper Alg.6)", "CLT of replicates, B=1000 (paper Alg.6)"]}
        wid = {k: 0.0 for k in hits}
        for _ in range(R):
            k = rng.binomial(n, theta)
            phat = k / n
            cis = {"Wald": A.binomial_ci(k, n, method="wald"), "Wilson": A.binomial_ci(k, n, method="wilson"),
                   "Clopper-Pearson": A.binomial_ci(k, n, method="clopper-pearson")}
            reps = rng.binomial(n, phat, B) / n          # bootstrap replicates of accuracy
            cis["Bootstrap percentile"] = A.bootstrap_interval(reps, phat, method="percentile")
            cis["Bootstrap basic (paper Alg.7)"] = A.bootstrap_interval(reps, phat, method="basic")
            cis["CLT of replicates, B=30 (paper Alg.6)"] = A.clt_ci(reps[:30])
            cis["CLT of replicates, B=1000 (paper Alg.6)"] = A.clt_ci(reps)
            for name, ci in cis.items():
                hits[name] += ci.lower <= theta <= ci.upper
                wid[name] += ci.width
        for name in hits:
            rows.append({"theta": theta, "n": n, "method": name, "coverage": hits[name] / R, "mean_width": wid[name] / R})
df = pd.DataFrame(rows); df.to_csv(f"{RES}/e1_ci_coverage.csv", index=False)
print(df[(df.theta == .9)].pivot(index="method", columns="n", values="coverage").round(3))
# figure: coverage vs n at theta=.9 and .95
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
for ax, th in zip(axs, (0.90, 0.95)):
    for i, (m, g) in enumerate(df[df.theta == th].groupby("method", sort=False)):
        ax.plot(g.n, g.coverage, marker="o", ms=3, label=m, color=PAL[i % len(PAL)])
    ax.axhline(.95, color="k", lw=.8, ls="--"); ax.set_xscale("log"); ax.set_title(f"true accuracy = {th}")
    ax.set_xlabel("test-set size n")
axs[0].set_ylabel("empirical coverage (nominal 0.95)"); axs[0].set_ylim(0, 1.02)
axs[1].legend(fontsize=6, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/e1_coverage.png")
