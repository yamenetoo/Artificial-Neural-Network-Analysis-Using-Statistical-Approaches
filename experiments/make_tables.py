import json, pandas as pd, numpy as np
from common import RES
T = lambda s, name: open(f"{RES}/{name}", "w").write(s)
# ---- E1
d = pd.read_csv(f"{RES}/e1_ci_coverage.csv")
order = ["Wald", "Wilson", "Clopper-Pearson", "Bootstrap percentile", "Bootstrap basic (paper Alg.7)", "CLT of replicates, B=30 (paper Alg.6)", "CLT of replicates, B=1000 (paper Alg.6)"]
nice = {"Bootstrap basic (paper Alg.7)": "Bootstrap, basic (Alg.~7 of \\cite{almohamad2024})", "CLT of replicates, B=30 (paper Alg.6)": "CLT of replicates, $B=30$ (Alg.~6)",
        "CLT of replicates, B=1000 (paper Alg.6)": "CLT of replicates, $B=1000$ (Alg.~6)"}
def e1(theta):
    g = d[d.theta == theta].pivot(index="method", columns="n", values="coverage").loc[order]
    w = d[(d.theta == theta) & (d.n == 100)].set_index("method").mean_width.loc[order]
    s = "\\begin{tabular}{lrrrrr}\\toprule\nMethod & $n=30$ & $n=100$ & $n=300$ & $n=1000$ & width ($n{=}100$)\\\\\\midrule\n"
    for m in order:
        s += f"{nice.get(m, m)} & " + " & ".join(f"{x:.3f}" for x in g.loc[m]) + f" & {w[m]:.3f}\\\\\n"
    return s + "\\bottomrule\\end{tabular}\n"
T(e1(0.90), "t_e1_090.tex"); T(e1(0.99), "t_e1_099.tex")
# ---- E2
e2 = pd.read_csv(f"{RES}/e2_bootstrap_schemes.csv")
s = "\\begin{tabular}{llrr}\\toprule\nData set & Quantity & Estimate & 95\\% interval\\\\\\midrule\n"
for ds, g in e2.groupby("dataset", sort=False):
    for _, r in g.iterrows():
        s += f"{ds.replace('_', ' ')} & {r.scheme} & {r['mean']:.3f} & [{r.ci_low:.3f}, {r.ci_high:.3f}]\\\\\n"
    s += "\\midrule\n"
s = s.rsplit("\\midrule\n", 1)[0] + "\\bottomrule\\end{tabular}\n"; T(s, "t_e2.tex")
# ---- E3
e3 = pd.read_csv(f"{RES}/e3_multiplicity.csv")
s = "\\begin{tabular}{rrrrrrr}\\toprule\n$n$ & P(any FP) raw & P(any FP) Holm & FDR raw & FDR BH & power raw & power BH\\\\\\midrule\n"
for _, r in e3.iterrows():
    s += f"{int(r.n)} & {r.iloc[1]:.3f} & {r.iloc[2]:.3f} & {r.iloc[3]:.3f} & {r.iloc[4]:.3f} & {r.iloc[5]:.3f} & {r.iloc[6]:.3f}\\\\\n"
T(s + "\\bottomrule\\end{tabular}\n", "t_e3.tex")
# ---- E4
raw = pd.read_csv(f"{RES}/e4_reduction_raw.csv")
full = raw[raw.method == "full"].set_index(["dataset", "seed"])["metric"]
o = raw[raw.method != "full"].copy(); o["frac"] = (o.k_requested / o.width).round(2)
o["delta"] = [m - full[(ds, sd)] for m, ds, sd in zip(o.metric, o.dataset, o.seed)]
lab = {"lsc_qr": "LSC-QR", "lsc_medoid": "LSC-medoid", "lsc_importance": "LSC-importance", "prune_only": "prune w/o compensation", "cluster_mean": "cluster-mean$^\\dagger$", "pca": "PCA$^\\dagger$"}
for frac, name in ((0.5, "t_e4_050.tex"), (0.25, "t_e4_025.tex")):
    g = o[o.frac == frac]
    dss = ["iris", "wine", "breast_cancer", "digits", "diabetes"]
    s = "\\begin{tabular}{l" + "r" * 5 + "}\\toprule\nMethod & iris & wine & breast cancer & digits & diabetes ($R^2$)\\\\\\midrule\n"
    for m in lab:
        cells = []
        for ds in dss:
            x = g[(g.method == m) & (g.dataset == ds)].delta
            cells.append(f"{x.mean()*100:+.1f} $\\pm$ {x.std()*100:.1f}")
        s += f"{lab[m]} & " + " & ".join(cells) + "\\\\\n"
    base = [f"{full[ds].mean():.3f}" for ds in dss]
    s += "\\midrule\nfull network (metric) & " + " & ".join(base) + "\\\\\n\\bottomrule\\end{tabular}\n"
    T(s, name)
# ---- E5
c = json.load(open(f"{RES}/e5_case_studies.json"))
rows = [("Test accuracy (Wilson 95\\% CI)", lambda v: f"{v['accuracy']:.3f} [{v['wilson'][0]:.3f}, {v['wilson'][1]:.3f}]"),
        ("Bootstrap percentile 95\\% CI", lambda v: f"[{v['boot'][0]:.3f}, {v['boot'][1]:.3f}]"),
        ("Effective rank of hidden layer", lambda v: f"{v['effective_rank']:.2f} of {v['width']}"),
        ("Neurons associated with response (BH)", lambda v: f"{v['sep_sig']} / {v['width']}"),
        ("Neurons with significant ablation effect", lambda v: f"{v['abl_sig']} / {v['width']}"),
        ("Pairwise contrasts: Mann--Whitney raw / Holm", lambda v: f"{v['mwu_raw']} / {v['mwu_holm']} of {v['n_pairs']}" if 'n_pairs' in v else "n/a (binary)"),
        ("Pairwise contrasts: Wilcoxon raw / Holm", lambda v: f"{v['wil_raw']} / {v['wil_holm']} of {v['n_pairs']}" if 'n_pairs' in v else "n/a (binary)"),
        ("Surrogate McFadden $R^2$", lambda v: f"{v['surrogate']['mcfadden_r2']:.3f}"),
        ("Surrogate LR test, whole layer ($p$)", lambda v: f"{v['surrogate']['lr_test']['p_value']:.1e}"),
        ("Neurons significant, Wald ($p<.05$, raw)", lambda v: f"{v['surrogate_sig_coef_raw']} / {v['surrogate_n_coef'] if 'n_pairs' not in v else v['width']}"),
        ("Neurons significant, drop-one LR (BH)", lambda v: f"{v['drop_one_lr_sig']} / {v['drop_one_lr_n']}"),
        ("Surrogate--network agreement (test)", lambda v: f"{v['surrogate_fidelity']['agreement']:.3f}")]
s = "\\begin{tabular}{lrr}\\toprule\n & Iris ($k{=}10$) & Breast cancer ($k{=}16$)\\\\\\midrule\n"
for name, f in rows:
    s += f"{name} & {f(c['iris'])} & {f(c['breast_cancer'])}\\\\\n"
T(s + "\\bottomrule\\end{tabular}\n", "t_e5.tex")
print("tables ok")
