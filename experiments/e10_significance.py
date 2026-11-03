"""E10: formal comparison of pruning methods across data sets/models (Friedman, Nemenyi CD, Wilcoxon-Holm)."""
import os
import numpy as np, pandas as pd
from common import *
import annstat as A
SUB = ["lsc_qr", "obs", "thinet", "merge", "lsc_leverage", "lsc_random", "weight_norm"]   # methods that reduce the neurons computed
LAB = {"lsc_qr": "LSC-QR", "obs": "OBS-greedy", "thinet": "ThiNet", "merge": "Neuron merging", "lsc_leverage": "CUR+LSC",
       "lsc_random": "random+LSC", "weight_norm": "weight-norm"}
frames = []
def add(path, group):
    if not os.path.exists(path): return
    o = pd.read_csv(path); o = o.rename(columns={group: "block"})
    full = o[o.method == "full"].set_index(["block", "seed"])["metric"]
    d = o[o.method != "full"].copy(); d["frac"] = (d.k_requested / d.width).round(3)
    d["delta"] = d.apply(lambda r: 100 * (r.metric - full[(r.block, r.seed)]), axis=1)
    frames.append(d[["block", "seed", "method", "frac", "delta"]])
add(f"{RES}/e6_baselines_raw.csv", "dataset"); add(f"{RES}/e7_torch_raw.csv", "model"); add(f"{RES}/e9_cifar_raw.csv", "model")
L = pd.concat(frames); L = L[L.method.isin(SUB)]
blocks = sorted(L.block.unique()); print("blocks:", blocks)
res = {}; pair_rows = []; within_rows = []
for frac in (0.5, 0.25):
    sub = L[np.isclose(L.frac, frac)]
    scores = sub.groupby(["block", "method"]).delta.mean().unstack()[SUB]
    r = A.compare_methods(scores, control="lsc_qr", higher_is_better=True)
    res[frac] = r
    pw = r["pairwise"].copy(); pw["frac"] = frac; pair_rows.append(pw)
    w = A.paired_seed_tests(sub, "block", "method", "seed", "delta", control="lsc_qr")
    w["frac"] = frac; within_rows.append(w)
pairs = pd.concat(pair_rows); within = pd.concat(within_rows)
pairs.to_csv(f"{RES}/e10_pairs.csv", index=False); within.to_csv(f"{RES}/e10_within.csv", index=False)
# ---- tables
head = "\\begin{tabular}{lrr}\\toprule\nMethod & mean rank (50\\%) & mean rank (25\\%)\\\\\\midrule\n"
ranks = pd.DataFrame({f: res[f]["friedman"]["mean_rank"] for f in (0.5, 0.25)}).loc[SUB]
body = "\n".join(f"{LAB[m]} & {ranks.loc[m,0.5]:.2f} & {ranks.loc[m,0.25]:.2f}\\\\" for m in SUB)
f5, f25 = res[0.5]["friedman"], res[0.25]["friedman"]
foot = (f"\\midrule\nFriedman $\\chi^2$ ($p$) & {f5['chi2']:.1f} ({f5['p_chi2']:.1e}) & {f25['chi2']:.1f} ({f25['p_chi2']:.1e})\\\\\n"
        f"Iman--Davenport $F$ ($p$) & {f5['F_iman_davenport']:.1f} ({f5['p_F']:.1e}) & {f25['F_iman_davenport']:.1f} ({f25['p_F']:.1e})\\\\\n"
        f"Nemenyi CD ($\\alpha{{=}}0.05$) & {res[0.5]['cd']:.2f} & {res[0.25]['cd']:.2f}\\\\\n\\bottomrule\\end{{tabular}}\n")
open(f"{RES}/t_e10_ranks.tex", "w").write(head + body + "\n" + foot)
head = "\\begin{tabular}{lrrrrrr}\\toprule\n & \\multicolumn{3}{c}{50\\% width} & \\multicolumn{3}{c}{25\\% width}\\\\\nLSC-QR vs & wins/losses & median diff & $p_{\\mathrm{Holm}}$ & wins/losses & median diff & $p_{\\mathrm{Holm}}$\\\\\\midrule\n"
lines = []
for m in SUB[1:]:
    cells = []
    for f in (0.5, 0.25):
        r = pairs[(pairs.b == m) & np.isclose(pairs.frac, f)].iloc[0]
        cells += [f"{r.wins_a}/{r.wins_b}", f"{r.median_diff:+.2f}", f"{r.p_adj:.3f}"]
    lines.append(LAB[m] + " & " + " & ".join(cells) + "\\\\")
open(f"{RES}/t_e10_pairs.tex", "w").write(head + "\n".join(lines) + f"\n\\bottomrule\\end{{tabular}}\n")
# within-block counts (blocks where LSC-QR significantly better/worse after Holm over 6 comparisons)
hd = "\\begin{tabular}{lrrrr}\\toprule\n & \\multicolumn{2}{c}{50\\% width} & \\multicolumn{2}{c}{25\\% width}\\\\\nLSC-QR vs & better & worse & better & worse\\\\\\midrule\n"
ln = []
for m in SUB[1:]:
    cells = []
    for f in (0.5, 0.25):
        w = within[(within.b == m) & np.isclose(within.frac, f)]
        cells += [str(int(((w.mean_diff > 0) & w.significant).sum())), str(int(((w.mean_diff < 0) & w.significant).sum()))]
    ln.append(LAB[m] + " & " + " & ".join(cells) + "\\\\")
nb = len(blocks)
open(f"{RES}/t_e10_within.tex", "w").write(hd + "\n".join(ln) + f"\n\\bottomrule\\end{{tabular}}\n")
print(ranks.round(2)); print(f5['chi2'], f5['p_chi2'], f25['chi2'], f25['p_chi2'], res[0.5]['cd'], res[0.25]['cd'])
print(pairs[["frac","b","wins_a","wins_b","median_diff","p_adj"]].round(3).to_string())
print(open(f"{RES}/t_e10_within.tex").read())
