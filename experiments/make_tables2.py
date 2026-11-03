"""LaTeX tables for E6 (baselines), E7 (PyTorch) and E8 (bound)."""
import pandas as pd, numpy as np
from common import RES
NAMES = {"lsc_qr": "LSC-QR (label-free)", "obs": "OBS-greedy \\cite{hassibi1993,dong2017}", "thinet": "ThiNet \\cite{luo2017}",
         "merge": "Neuron merging, data-free \\cite{srinivas2015,kim2020}", "lsc_leverage": "CUR leverage + LSC \\cite{mahoney2009}",
         "lsc_random": "random subset + LSC", "weight_norm": "weight-norm pruning \\cite{han2015}",
         "cluster_mean": "cluster-mean$^\\dagger$", "pca": "PCA$^\\dagger$"}
ORDER = list(NAMES)
def delta(path, group):
    o = pd.read_csv(path)
    full = o[o.method == "full"].set_index([group, "seed"])["metric"]
    d = o[o.method != "full"].copy(); d["frac"] = (d.k_requested / d.width).round(2)
    d["delta"] = d.apply(lambda r: 100 * (r.metric - full[(r[group], r.seed)]), axis=1)
    return d, full
def fmt(x, s): return f"{x:+.1f} $\\pm$ {s:.1f}"
def table(d, group, cols, frac, bold=True):
    sub = d[d.frac == frac]
    g = sub.groupby(["method", group]).delta.agg(["mean", "std"])
    lines = []
    best = {c: max(g.loc[(m, c), "mean"] for m in ORDER if (m, c) in g.index and not NAMES[m].endswith("$^\\dagger$")) for c in cols}
    for m in ORDER:
        cells = []
        for c in cols:
            mu, sd = g.loc[(m, c)]
            t = fmt(mu, sd)
            if bold and abs(mu - best[c]) < 1e-9 and not NAMES[m].endswith("$^\\dagger$"): t = "\\textbf{" + t + "}"
            cells.append(t)
        lines.append(NAMES[m] + " & " + " & ".join(cells) + "\\\\")
    return lines
# ---- E6
d, full = delta(f"{RES}/e6_baselines_raw.csv", "dataset")
cols = ["iris", "wine", "breast_cancer", "digits", "diabetes"]
head = "\\begin{tabular}{lrrrrr}\\toprule\nMethod & iris & wine & breast cancer & digits & diabetes ($R^2$)\\\\\\midrule\n"
for frac, tag in ((.5, "050"), (.25, "025")):
    L = table(d, "dataset", cols, frac)
    ff = full.groupby("dataset").mean()
    L.append("\\midrule")
    L.append("full network (metric) & " + " & ".join(f"{ff[c]:.3f}" for c in cols) + "\\\\")
    open(f"{RES}/t_e6_{tag}.tex", "w").write(head + "\n".join(L) + "\n\\bottomrule\\end{tabular}\n")
# ---- E7
try:
    d7, full7 = delta(f"{RES}/e7_torch_raw.csv", "model")
    cols7 = ["mlp3", "cnn"]
    head7 = "\\begin{tabular}{lrrrr}\\toprule\n & \\multicolumn{2}{c}{deep MLP ($k{=}64$)} & \\multicolumn{2}{c}{CNN ($k{=}128$)}\\\\\nMethod & 50\\% width & 25\\% width & 50\\% width & 25\\% width\\\\\\midrule\n"
    g = d7.groupby(["method", "model", "frac"]).delta.agg(["mean", "std"])
    best = {(mo, fr): max(g.loc[(m, mo, fr), "mean"] for m in ORDER if not NAMES[m].endswith("$^\\dagger$")) for mo in cols7 for fr in (.5, .25)}
    L = []
    for m in ORDER:
        cells = []
        for mo in cols7:
            for fr in (.5, .25):
                mu, sd = g.loc[(m, mo, fr)]
                t = fmt(mu, sd)
                if abs(mu - best[(mo, fr)]) < 1e-9 and not NAMES[m].endswith("$^\\dagger$"): t = "\\textbf{" + t + "}"
                cells.append(t)
        L.append(NAMES[m] + " & " + " & ".join(cells) + "\\\\")
    ff = full7.groupby("model").mean()
    L.append("\\midrule"); L.append(f"full network (test accuracy) & \\multicolumn{{2}}{{c}}{{{ff['mlp3']:.3f}}} & \\multicolumn{{2}}{{c}}{{{ff['cnn']:.3f}}}\\\\")
    open(f"{RES}/t_e7.tex", "w").write(head7 + "\n".join(L) + "\n\\bottomrule\\end{tabular}\n")
    au = pd.read_csv(f"{RES}/e7_torch_audit.csv").groupby("model").mean(numeric_only=True)
    rows = ["Hidden units $k$ & " + " & ".join(f"{int(au.loc[m,'k'])}" for m in ("mlp3", "cnn")) + "\\\\",
            "Test accuracy & " + " & ".join(f"{au.loc[m,'test_acc']:.3f}" for m in ("mlp3", "cnn")) + "\\\\",
            "Effective rank & " + " & ".join(f"{au.loc[m,'effective_rank']:.1f}" for m in ("mlp3", "cnn")) + "\\\\",
            "Participation ratio & " + " & ".join(f"{au.loc[m,'participation_ratio']:.1f}" for m in ("mlp3", "cnn")) + "\\\\",
            "Dead neurons & " + " & ".join(f"{au.loc[m,'dead']:.1f}" for m in ("mlp3", "cnn")) + "\\\\",
            "Neurons associated with class (BH) & " + " & ".join(f"{au.loc[m,'assoc_sig']:.1f}" for m in ("mlp3", "cnn")) + "\\\\",
            "Neurons with significant ablation effect (BH) & " + " & ".join(f"{au.loc[m,'ablation_sig']:.1f}" for m in ("mlp3", "cnn")) + "\\\\"]
    open(f"{RES}/t_e7_audit.tex", "w").write("\\begin{tabular}{lrr}\\toprule\n & deep MLP & CNN\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
except FileNotFoundError:
    print("no E7 yet")
# ---- E8
b = pd.read_csv(f"{RES}/e8_bound.csv")
b["ok"] = b.train_mse <= b.bound * (1 + 1e-6); b["tt"] = b.test_mse / b.train_mse; b["qo"] = b.qr_resid / b.opt_tail; b["rel"] = b.test_mse / b.logit_var
g = b.groupby(["dataset", "frac"]).agg(ok=("ok", "mean"), tt=("tt", "median"), qo=("qo", "median"), rel=("rel", "median"), slack=("bound", "median"), tr=("train_mse", "median")).reset_index()
g["slackr"] = g.slack / g.tr
L = [f"{r.dataset.replace('_',' ')} & {int(r.frac*100)}\\% & {100*r.ok:.0f}\\% & {r.slackr:.1f} & {r.qo:.2f} & {r.tt:.2f} & {r.rel:.3f}\\\\" for r in g.itertuples()]
open(f"{RES}/t_e8.tex", "w").write("\\begin{tabular}{llrrrrr}\\toprule\nData & width & bound holds & bound/actual & QR/optimal & test/train & rel.\\ test error\\\\\\midrule\n" + "\n".join(L) + "\n\\bottomrule\\end{tabular}\n")
print("ok")
# ---- E9 (CIFAR-10)
import os
if os.path.exists(f"{RES}/e9_cifar_raw.csv"):
    d9, full9 = delta(f"{RES}/e9_cifar_raw.csv", "model")
    g = d9.groupby(["method", "frac"]).delta.agg(["mean", "std"])
    fr = (.5, .25, .125)
    best = {f: max(g.loc[(m, f), "mean"] for m in ORDER if m in d9.method.unique() and not NAMES[m].endswith("$^\\dagger$")) for f in fr}
    L = []
    for m in ORDER:
        if m not in d9.method.unique(): continue
        cells = []
        for f in fr:
            mu, sd = g.loc[(m, f)]; t = fmt(mu, sd)
            if abs(mu - best[f]) < 1e-9: t = "\\textbf{" + t + "}"
            cells.append(t)
        L.append(NAMES[m] + " & " + " & ".join(cells) + "\\\\")
    L.append("\\midrule"); L.append(f"full network (test accuracy) & \\multicolumn{{3}}{{c}}{{{full9.mean():.3f}}}\\\\")
    open(f"{RES}/t_e9.tex", "w").write("\\begin{tabular}{lrrr}\\toprule\nMethod & 50\\% ($m{=}128$) & 25\\% ($m{=}64$) & 12.5\\% ($m{=}32$)\\\\\\midrule\n" + "\n".join(L) + "\n\\bottomrule\\end{tabular}\n")
    a9 = pd.read_csv(f"{RES}/e9_cifar_audit.csv").mean(numeric_only=True)
    open(f"{RES}/t_e9_audit.tex", "w").write("\\begin{tabular}{lr}\\toprule\n & CIFAR-10 CNN\\\\\\midrule\n" +
        f"Hidden units $k$ & 256\\\\\nTest accuracy & {a9.test_acc:.3f}\\\\\nEffective rank & {a9.effective_rank:.1f}\\\\\nParticipation ratio & {a9.participation_ratio:.1f}\\\\\n"
        f"Dead neurons & {a9.dead:.1f}\\\\\nNeurons associated with class (BH) & {a9.assoc_sig:.1f}\\\\\nNeurons with significant ablation effect (BH) & {a9.ablation_sig:.1f}\\\\\n\\bottomrule\\end{tabular}\n")
