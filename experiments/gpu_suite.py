"""annstat GPU experiment suite (E1, E1b, E2, E3, E5, E6, E6b, E7).

Designed for Google Colab (GPU runtime) but works anywhere with PyTorch.
Every experiment writes its CSV (and figure) to OUT right after finishing, and finished
experiments are skipped on re-run, so an interrupted session can simply be restarted.

Configuration through environment variables (all optional):
    ANNSTAT_PRESET  quick | full           (default quick)
    ANNSTAT_OUT     output directory       (default /content/drive/MyDrive/annstat_results, or ./annstat_results off Colab)
    ANNSTAT_DATA    dataset cache          (default /content/data or ./data)
    ANNSTAT_RUN     comma list of tags     (default E1,E1b,E3,E5,E6,E6b,E2,E7)
    ANNSTAT_FORCE   1 to recompute         (default 0)

E1   coverage of accuracy intervals (GPU Monte-Carlo, extended grid + figure)
E1b  exact coverage law of the CLT-of-replicates interval vs B (Proposition 1)
E2   retraining-bootstrap schemes (own-sample / warm OOB / cold OOB) on MNIST
E3   multiplicity control (raw/Bonferroni/Holm/BH/BY) with correlated neurons
E5   effective rank / ablation tests / compensated pruning vs width and depth
E6   compression baselines on MNIST, Fashion-MNIST, CIFAR-10 (+ error-bound check)
E6b  the same baselines on tabular data (sklearn MLPs, CPU)
E7   Friedman / Nemenyi / Wilcoxon-Holm across all blocks
"""
import os, sys, json, importlib, copy, time, traceback, warnings
warnings.filterwarnings("ignore")

ON_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if ON_COLAB and not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")

PRESET = os.environ.get("ANNSTAT_PRESET", "quick")
OUT = os.environ.get("ANNSTAT_OUT", "/content/drive/MyDrive/annstat_results" if ON_COLAB else "annstat_results")
DATA = os.environ.get("ANNSTAT_DATA", "/content/data" if ON_COLAB else "data")
RUN = os.environ.get("ANNSTAT_RUN", "E1,E1b,E3,E5,E6,E6b,E2,E7").split(",")
FORCE = os.environ.get("ANNSTAT_FORCE", "0") == "1"
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

# make the package importable straight from the repo checkout (no pip needed)
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
_src = os.path.join(os.path.dirname(_here), "src")
if os.path.exists(os.path.join(_src, "annstat", "__init__.py")) and _src not in sys.path:
    sys.path.insert(0, _src)
importlib.invalidate_caches()

P = {
 "quick": dict(R_E1=2000,  R_E1b=3000,  R_E3=300,  E5_ds=("mnist",), E5_widths=(16, 64, 256), E5_depths=(1, 2), E5_seeds=1,
               E5_ntrain=5000, E5_epochs=3, E6_ds=("mnist", "cifar10"), E6_seeds=2, E6_ntrain=6000, E6_epochs=3, E6_nfit=2000,
               E6_ntest=2000, tab_seeds=3, rt_B=6, rt_Bcold=3, rt_ntrain=3000, rt_base_epochs=4, rt_ft_epochs=1, rt_cold_epochs=3),
 "full":  dict(R_E1=20000, R_E1b=20000, R_E3=2000, E5_ds=("mnist", "fashion"), E5_widths=(16, 32, 64, 128, 256, 512), E5_depths=(1, 2, 3),
               E5_seeds=3, E5_ntrain=20000, E5_epochs=8, E6_ds=("mnist", "fashion", "cifar10"), E6_seeds=5, E6_ntrain=30000,
               E6_epochs=15, E6_nfit=6000, E6_ntest=10000, tab_seeds=15, rt_B=60, rt_Bcold=30, rt_ntrain=10000, rt_base_epochs=8,
               rt_ft_epochs=3, rt_cold_epochs=8),
}[PRESET]

import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy import stats
from scipy.linalg import qr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import annstat as A

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
torch.backends.cudnn.benchmark = True
print("annstat", A.__version__, "| device:", DEV, (torch.cuda.get_device_name(0) if DEV == "cuda" else "  (!!! no GPU: Runtime > Change runtime type > GPU)"))
print("preset:", PRESET, "| output:", OUT)

LOG = {"timings": {}, "errors": {}}
def have(name): return (not FORCE) and os.path.exists(f"{OUT}/{name}.csv")
def save(df, name): df.to_csv(f"{OUT}/{name}.csv", index=False); print(f"   saved {name}.csv  ({len(df)} rows)")
def run(tag, fn):
    if tag not in RUN: return
    t = time.time(); print(f"\n===== {tag} =====")
    try:
        fn(); LOG["timings"][tag] = round(time.time() - t, 1)
    except Exception as e:
        LOG["errors"][tag] = traceback.format_exc(); print(f"!! {tag} failed: {e}\n{LOG['errors'][tag][-1500:]}")
    print(f"   {tag}: {time.time()-t:.0f}s")

# =============================== E1 / E1b : interval coverage (GPU) ===============================
def cp_table(n, level=0.95):
    k = np.arange(n + 1); a = 1 - level
    with np.errstate(all="ignore"):
        lo = np.where(k == 0, 0.0, stats.beta.ppf(a / 2, k, n - k + 1))
        hi = np.where(k == n, 1.0, stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return torch.tensor(lo, device=DEV, dtype=DT), torch.tensor(hi, device=DEV, dtype=DT)

def e1_cell(theta, n, R, Bs, level=0.95, chunk=2000, Bboot=1000):
    a = 1 - level; z = stats.norm.ppf(1 - a / 2)
    lo_cp, hi_cp = cp_table(n, level)
    tcrit = {B: float(stats.t.ppf(1 - a / 2, B - 1)) for B in Bs}
    Bmax = max(max(Bs), Bboot); acc_ = {}
    def add(name, lo, hi):
        s = acc_.setdefault(name, [0.0, 0.0, 0])
        s[0] += ((lo <= theta) & (theta <= hi)).double().sum().item(); s[1] += (hi - lo).sum().item(); s[2] += len(lo)
    qs = torch.tensor([a / 2, 1 - a / 2], device=DEV, dtype=DT)
    for st in range(0, R, chunk):
        r = min(chunk, R - st)
        k = torch.binomial(torch.full((r,), float(n), device=DEV, dtype=DT), torch.full((r,), float(theta), device=DEV, dtype=DT))
        p = k / n; se = torch.sqrt(p * (1 - p) / n)
        add("Wald", p - z * se, p + z * se)
        c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * torch.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        add("Wilson", c - h, c + h)
        ki = k.long(); add("Clopper-Pearson", lo_cp[ki], hi_cp[ki])
        reps = torch.binomial(torch.full((r, Bmax), float(n), device=DEV, dtype=DT), p[:, None].expand(r, Bmax).contiguous()) / n
        q = torch.quantile(reps[:, :Bboot], qs, dim=1)
        add("Bootstrap percentile", q[0], q[1]); add("Bootstrap basic", 2 * p - q[1], 2 * p - q[0])
        for B in Bs:
            m = reps[:, :B].mean(1); s = reps[:, :B].std(1); hw = tcrit[B] * s / np.sqrt(B)
            add(f"CLT of replicates B={B}", m - hw, m + hw)
    return [dict(theta=theta, n=n, method=nm, coverage=v[0] / v[2], mean_width=v[1] / v[2]) for nm, v in acc_.items()]

def exp_E1():
    if have("e1_ci_coverage"): print("   (exists, skipped)"); return
    torch.manual_seed(0); rows = []
    for theta in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        for n in (20, 30, 50, 100, 200, 300, 500, 1000):
            rows += e1_cell(theta, n, P["R_E1"], Bs=(30, 100, 1000))
    e1 = pd.DataFrame(rows); save(e1, "e1_ci_coverage")
    print(e1[e1.theta == 0.9].pivot(index="method", columns="n", values="coverage").round(3).to_string())
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, th in zip(axs, (0.8, 0.9, 0.95)):
        for m, g in e1[e1.theta == th].groupby("method"): ax.plot(g.n, g.coverage, marker="o", ms=3, label=m, ls=":" if "CLT" in m else "-")
        ax.axhline(0.95, color="k", ls="--", lw=.8); ax.set_xscale("log"); ax.set_title(f"theta={th}"); ax.set_xlabel("n")
    axs[0].set_ylabel("coverage"); axs[2].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{OUT}/e1_coverage.png", dpi=150); plt.close(fig)

def exp_E1b():
    if have("e1b_clt_law"): print("   (exists, skipped)"); return
    torch.manual_seed(1); Bs = (2, 3, 5, 10, 20, 30, 50, 100, 300, 1000); rows = []
    for theta, n in ((0.9, 5000), (0.5, 5000), (0.8, 20000)):
        for r in e1_cell(theta, n, P["R_E1b"], Bs=Bs):
            if r["method"].startswith("CLT"):
                B = int(r["method"].split("=")[1]); t = stats.t.ppf(0.975, B - 1)
                rows.append(dict(theta=theta, n=n, B=B, empirical=r["coverage"], theory=2 * stats.t.cdf(t / np.sqrt(B + 1), B - 1) - 1))
    d = pd.DataFrame(rows); d["abs_err"] = (d.empirical - d.theory).abs(); save(d, "e1b_clt_law")
    print(d.round(3).to_string(index=False))

# =============================== E3 : multiplicity (GPU) ===============================
def adj_np(p, method):
    r, k = p.shape; o = np.argsort(p, 1); ps = np.take_along_axis(p, o, 1); i = np.arange(1, k + 1)
    if method == "bonferroni": return np.minimum(p * k, 1.0)
    if method == "holm": q = np.minimum(np.maximum.accumulate((k - i + 1) * ps, 1), 1.0)
    else:
        c = np.sum(1.0 / i) if method == "by" else 1.0
        q = np.minimum(np.minimum.accumulate((ps * k / i * c)[:, ::-1], 1)[:, ::-1], 1.0)
    out = np.empty_like(q); np.put_along_axis(out, o, q, 1); return out

def e3_cell(n, k, rho, R, shift=0.5, q=3, frac=0.2, alpha=0.05):
    k_rel = int(round(k * frac)); null = np.arange(k) >= k_rel
    methods = ("raw", "bonferroni", "holm", "bh", "by"); acc_ = {m: [0, 0.0, 0.0, 0] for m in methods}
    chunk = max(20, int(2e7 / (n * k)))
    sig = torch.zeros(k, device=DEV, dtype=DT); sig[:k_rel] = shift
    for st in range(0, R, chunk):
        r = min(chunk, R - st)
        y = torch.randint(0, q, (r, n), device=DEV)
        X = np.sqrt(1 - rho) * torch.randn(r, n, k, device=DEV, dtype=DT) + np.sqrt(rho) * torch.randn(r, n, 1, device=DEV, dtype=DT)
        X = X + y[:, :, None].to(DT) * sig[None, None, :]
        ranks = X.argsort(1).argsort(1).to(DT) + 1
        oh = torch.nn.functional.one_hot(y, q).to(DT)
        Rc = torch.einsum("rnk,rnc->rck", ranks, oh); nc = oh.sum(1).clamp(min=1)
        H = 12 / (n * (n + 1)) * (Rc ** 2 / nc[:, :, None]).sum(1) - 3 * (n + 1)
        p = stats.chi2.sf(H.cpu().numpy(), q - 1)
        for m in methods:
            rej = (p if m == "raw" else adj_np(p, m)) < alpha
            V = (rej & null).sum(1); Rr = rej.sum(1); a = acc_[m]
            a[0] += int((V > 0).sum()); a[1] += float((V / np.maximum(Rr, 1)).sum()); a[2] += float(rej[:, ~null].mean(1).sum()); a[3] += r
    return [dict(n=n, k=k, rho=rho, method=m, FWER=v[0] / v[3], FDR=v[1] / v[3], power=v[2] / v[3]) for m, v in acc_.items()]

def exp_E3():
    if have("e3_multiplicity"): print("   (exists, skipped)"); return
    torch.manual_seed(7); rows = []
    for n in (60, 150, 300):
        for k in (20, 50, 200):
            for rho in (0.0, 0.5, 0.8):
                rows += e3_cell(n, k, rho, P["R_E3"])
    d = pd.DataFrame(rows); save(d, "e3_multiplicity")
    print(d[(d.k == 50) & (d.rho == 0)].round(3).to_string(index=False))
    print("\nFDR of BH vs BY under dependence (k=50):")
    print(d[(d.k == 50) & d.method.isin(["bh", "by"])].pivot_table(index=["n", "rho"], columns="method", values="FDR").round(3).to_string())

# =============================== data + networks (GPU) ===============================
_DS = {}
def DS(name):
    if name in _DS: return _DS[name]
    import torchvision as tv
    cls = {"mnist": tv.datasets.MNIST, "fashion": tv.datasets.FashionMNIST, "cifar10": tv.datasets.CIFAR10}[name]
    tr, te = cls(DATA, train=True, download=True), cls(DATA, train=False, download=True)
    def conv(ds):
        x = torch.as_tensor(np.asarray(ds.data)).float() / 255
        x = x[:, None] if x.ndim == 3 else x.permute(0, 3, 1, 2)
        return x, torch.as_tensor(np.asarray(ds.targets)).long()
    (xtr, ytr), (xte, yte) = conv(tr), conv(te)
    m, s = xtr.mean((0, 2, 3), keepdim=True), xtr.std((0, 2, 3), keepdim=True)
    _DS[name] = ((xtr - m) / s, ytr, (xte - m) / s, yte); return _DS[name]

def build_mlp(in_dim, sizes):
    L = [nn.Flatten()]; prev = in_dim
    for s in sizes: L += [nn.Linear(prev, s), nn.ReLU()]; prev = s
    L.append(nn.Linear(prev, 10)); return nn.Sequential(*L)

def build_cnn(in_ch, width=256, side=32):
    s = side // 8
    return nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                         nn.Linear(128 * s * s, width), nn.ReLU(), nn.Linear(width, 10))

def parts(net):   # (net, last-hidden ReLU, head Linear, last hidden Linear)
    return net, net[-2], net[-1], net[-3]

def train(net, X, y, epochs, bs=256, lr=1e-3, wd=1e-5):
    net.to(DEV).train(); X, y = X.to(DEV), y.to(DEV); opt = torch.optim.Adam(net.parameters(), lr, weight_decay=wd); n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; opt.zero_grad(); nn.functional.cross_entropy(net(X[b]), y[b]).backward(); opt.step()
    return net.eval()

@torch.no_grad()
def acc(net, X, y):
    net.eval(); X, y = X.to(DEV), y.to(DEV)
    return float(torch.cat([net(X[i:i + 4096]).argmax(1) for i in range(0, len(y), 4096)]).eq(y).float().mean())

def make_adapter(net, hid, head):
    W = head.weight.detach().cpu().numpy().T.astype(float); b = head.bias.detach().cpu().numpy().astype(float); store = {}
    hid.register_forward_hook(lambda m, i, o: store.__setitem__("h", o.detach()))
    def hidden_fn(X):
        net.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(X), 4096):
                net(torch.as_tensor(X[s:s + 4096], dtype=torch.float32, device=DEV)); out.append(store["h"].cpu().numpy())
        return np.vstack(out).astype(float)
    return A.FunctionalAdapter(hidden_fn, W, b, classes=np.arange(10))

METHODS = ("lsc_qr", "obs", "thinet", "merge", "lsc_leverage", "lsc_random", "weight_norm", "cluster_mean", "pca")

# =============================== E5 : rank vs width / depth ===============================
def exp_E5():
    if have("e5_rank_vs_width"): print("   (exists, skipped)"); return
    rows = []
    for ds in P["E5_ds"]:
        Xtr, ytr, Xte, yte = DS(ds)
        for depth in P["E5_depths"]:
            for width in P["E5_widths"]:
                for seed in range(P["E5_seeds"]):
                    torch.manual_seed(seed); rng = np.random.default_rng(seed)
                    idx = torch.as_tensor(rng.choice(len(ytr), P["E5_ntrain"], replace=False))
                    net, hid, head, lin = parts(build_mlp(Xtr[0].numel(), [width] * depth))
                    train(net, Xtr[idx], ytr[idx], P["E5_epochs"]); a = make_adapter(net, hid, head)
                    Xe, ye = Xte[:2000].numpy(), yte[:2000].numpy(); Xf, yf = Xtr[idx][:3000].numpy(), ytr[idx][:3000].numpy()
                    rs = A.redundancy_summary(a.hidden(Xe)); abl = A.ablation_importance(a, Xe, ye, B=300, rng=0)
                    red = A.reduction_path(a, Xf, yf, Xe, ye, [max(2, width // 2), max(2, width // 4)], ("lsc_qr", "obs"))
                    full = red[red.method == "full"].metric.iloc[0]; r = {}
                    for m in ("lsc_qr", "obs"):
                        for f, kk in ((50, max(2, width // 2)), (25, max(2, width // 4))):
                            r[f"{m}_d{f}"] = 100 * (red[(red.method == m) & (red.k_requested == kk)].metric.iloc[0] - full)
                    rows.append(dict(dataset=ds, depth=depth, width=width, seed=seed, test_acc=acc(net, Xte[:5000], yte[:5000]),
                                     eff_rank=rs["effective_rank"], part_ratio=rs["participation_ratio"], dead=rs["n_dead"],
                                     n_ablation_sig=int(abl.significant.sum()), **r))
                    print(f"   {ds} depth={depth} width={width} seed={seed}: erank={rs['effective_rank']:.1f} abl_sig={int(abl.significant.sum())}", flush=True)
                    del net; torch.cuda.empty_cache()
    d = pd.DataFrame(rows); save(d, "e5_rank_vs_width")
    fig, axs = plt.subplots(1, len(P["E5_ds"]), figsize=(5 * len(P["E5_ds"]), 3.8), squeeze=False)
    for ax, ds in zip(axs[0], P["E5_ds"]):
        for dep, g in d[d.dataset == ds].groupby("depth"):
            g = g.groupby("width").eff_rank.mean(); ax.plot(g.index, g.values, marker="o", label=f"depth {dep}")
        ax.plot(P["E5_widths"], P["E5_widths"], "k--", lw=.7, label="rank = width"); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(ds); ax.set_xlabel("last hidden width k"); ax.set_ylabel("effective rank"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/e5_rank_vs_width.png", dpi=150); plt.close(fig)
    print(d.groupby(["dataset", "depth", "width"])[["eff_rank", "part_ratio", "dead", "n_ablation_sig", "lsc_qr_d50", "obs_d50", "lsc_qr_d25", "obs_d25"]].mean().round(2).to_string())

# =============================== E6 : baselines on image data + bound check ===============================
def exp_E6():
    if have("e6_gpu_raw"): print("   (exists, skipped)"); return
    rows, bound_rows = [], []
    for ds in P["E6_ds"]:
        Xtr, ytr, Xte, yte = DS(ds)
        for seed in range(P["E6_seeds"]):
            t0 = time.time(); torch.manual_seed(seed); rng = np.random.default_rng(seed)
            idx = torch.as_tensor(rng.choice(len(ytr), min(P["E6_ntrain"], len(ytr)), replace=False))
            raw = build_cnn(Xtr.shape[1], 256, Xtr.shape[-1]) if ds == "cifar10" else build_mlp(Xtr[0].numel(), [256, 128, 64])
            net, hid, head, lin = parts(raw); k = lin.out_features
            train(net, Xtr[idx], ytr[idx], P["E6_epochs"]); a = make_adapter(net, hid, head)
            fit = rng.choice(len(idx), min(P["E6_nfit"], len(idx)), replace=False)
            Xf, yf = Xtr[idx][fit].numpy(), ytr[idx][fit].numpy(); Xe, ye = Xte[:P["E6_ntest"]].numpy(), yte[:P["E6_ntest"]].numpy()
            inc = (lin.weight.detach().cpu().numpy().T.astype(float), lin.bias.detach().cpu().numpy().astype(float))
            t1 = time.time()
            df = A.reduction_path(a, Xf, yf, Xe, ye, [k // 2, k // 4, k // 8], METHODS, incoming=inc, seed=seed)
            print(f"     train+adapter {t1 - t0:.0f}s | reduction_path {time.time() - t1:.0f}s", flush=True)
            df["dataset"], df["seed"], df["width"] = f"{ds}_{'cnn' if ds == 'cifar10' else 'mlp3'}", seed, k; rows.append(df)
            H, He = a.hidden(Xf), a.hidden(Xe); W, b = a.head(); Z, Ze = H @ W + b, He @ W + b
            Hc = H - H.mean(0); _, Rq, _ = qr(Hc, pivoting=True, mode="economic")
            for fr in (2, 4, 8):
                m = k // fr; keep = A.select_qr(H, m); drop = np.setdiff1d(np.arange(k), keep); red = A.compress_lsc(H, W, b, keep)
                R22 = Rq[m:, m:]; sv = np.linalg.svd(Hc, compute_uv=False)
                bound_rows.append(dict(dataset=df.dataset.iloc[0], seed=seed, frac=1 / fr, m=m, train_mse=((Z - red.logits(H)) ** 2).sum(1).mean(),
                                       test_mse=((Ze - red.logits(He)) ** 2).sum(1).mean(),
                                       bound=np.linalg.norm(R22, "fro") ** 2 * np.linalg.norm(W[drop], 2) ** 2 / len(H),
                                       qr_resid=np.linalg.norm(R22, "fro") ** 2 / len(H), opt_tail=(sv[m:] ** 2).sum() / len(H), logit_var=Z.var(0).sum()))
            print(f"   {ds} seed {seed}: acc={acc(net, Xte[:5000], yte[:5000]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
            pd.concat(rows).to_csv(f"{OUT}/e6_gpu_raw_partial.csv", index=False); del net; torch.cuda.empty_cache()
    raw = pd.concat(rows); save(raw, "e6_gpu_raw"); b = pd.DataFrame(bound_rows); save(b, "e8_bound_gpu")
    full = raw[raw.method == "full"].set_index(["dataset", "seed"]).metric
    d = raw[raw.method != "full"].copy(); d["frac"] = (d.k_requested / d.width).round(3)
    d["delta"] = [100 * (m - full[(ds, s)]) for m, ds, s in zip(d.metric, d.dataset, d.seed)]
    print(d.groupby(["dataset", "frac", "method"]).delta.mean().unstack("method").round(2).to_string())
    print("\nbound holds (train error <= bound):", float((b.train_mse <= b.bound * (1 + 1e-6)).mean()),
          "| median test/train:", float((b.test_mse / b.train_mse).median()), "| median QR/optimal:", float((b.qr_resid / b.opt_tail).median()))

# =============================== E6b : baselines on tabular data (CPU) ===============================
def exp_E6b():
    if have("e6b_tab_raw"): print("   (exists, skipped)"); return
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler
    cfg = [("iris", load_iris, 12, "c"), ("wine", load_wine, 16, "c"), ("breast_cancer", load_breast_cancer, 16, "c"),
           ("digits", load_digits, 32, "c"), ("diabetes", load_diabetes, 16, "r")]
    rows = []
    for name, loader, w, kind in cfg:
        X, y = loader(return_X_y=True)
        for seed in range(P["tab_seeds"]):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=seed, stratify=y if kind == "c" else None)
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            m = (MLPClassifier((w,), max_iter=1500, random_state=seed) if kind == "c" else MLPRegressor((w,), max_iter=3000, random_state=seed, alpha=1e-2)).fit(Xtr, ytr)
            df = A.reduction_path(A.SklearnMLPAdapter(m), Xtr, ytr, Xte, yte, sorted({max(2, round(w * f)) for f in (.25, .5)}), METHODS,
                                  incoming=(m.coefs_[-2], m.intercepts_[-2]), seed=seed)
            df["dataset"], df["seed"], df["width"] = name, seed, w; rows.append(df)
        print("   ", name, "done", flush=True)
    save(pd.concat(rows), "e6b_tab_raw")

# =============================== E2 : retraining-bootstrap schemes at scale ===============================
def exp_E2():
    if have("e2_retrain_bootstrap"): print("   (exists, skipped)"); return
    Xtr, ytr, Xte, yte = DS("mnist"); rng = np.random.default_rng(0); torch.manual_seed(0)
    idx = torch.as_tensor(rng.choice(len(ytr), P["rt_ntrain"], replace=False)); Xp, yp = Xtr[idx].to(DEV), ytr[idx].to(DEV)
    Xt, yt = Xte[:5000].to(DEV), yte[:5000].to(DEV)
    base = train(build_mlp(784, [128]), Xp, yp, P["rt_base_epochs"]); hold = acc(base, Xt, yt)
    ci = A.binomial_ci(int(round(hold * len(yt))), len(yt)); own, oobw, oobc = [], [], []
    for b in range(P["rt_B"]):
        bi = torch.randint(0, len(yp), (len(yp),), device=DEV); inbag = torch.zeros(len(yp), dtype=torch.bool, device=DEV); inbag[bi] = True; oob = ~inbag
        m = train(copy.deepcopy(base), Xp[bi], yp[bi], P["rt_ft_epochs"])
        own.append(acc(m, Xp[bi], yp[bi])); oobw.append(acc(m, Xp[oob], yp[oob]))
        if b < P["rt_Bcold"]:
            mc = train(build_mlp(784, [128]), Xp[bi], yp[bi], P["rt_cold_epochs"]); oobc.append(acc(mc, Xp[oob], yp[oob]))
    rows = [dict(scheme=n, mean=np.mean(v), lo=np.quantile(v, .025), hi=np.quantile(v, .975), n_rep=len(v))
            for n, v in (("own-sample, warm start", own), ("out-of-bag, warm start", oobw), ("out-of-bag, cold start", oobc))]
    rows.append(dict(scheme="hold-out (Wilson)", mean=hold, lo=ci.lower, hi=ci.upper, n_rep=len(yt)))
    d = pd.DataFrame(rows); save(d, "e2_retrain_bootstrap"); print(d.round(4).to_string(index=False))

# =============================== E7 : formal significance tests ===============================
def exp_E7():
    SUB = ["lsc_qr", "obs", "thinet", "merge", "lsc_leverage", "lsc_random", "weight_norm"]; frames = []
    for f in ("e6_gpu_raw", "e6b_tab_raw"):
        if not os.path.exists(f"{OUT}/{f}.csv"): continue
        o = pd.read_csv(f"{OUT}/{f}.csv"); full = o[o.method == "full"].set_index(["dataset", "seed"]).metric
        d = o[o.method != "full"].copy(); d["frac"] = (d.k_requested / d.width).round(3)
        d["delta"] = [100 * (m - full[(b, s)]) for m, b, s in zip(d.metric, d.dataset, d.seed)]
        frames.append(d.rename(columns={"dataset": "block"})[["block", "seed", "method", "frac", "delta"]])
    L = pd.concat(frames); L = L[L.method.isin(SUB)]; print("blocks:", sorted(L.block.unique()))
    ranks, pairs, within = [], [], []
    for frac in (0.5, 0.25):
        sub = L[np.isclose(L.frac, frac)]; scores = sub.groupby(["block", "method"]).delta.mean().unstack()[SUB].dropna()
        r = A.compare_methods(scores, control="lsc_qr")
        fr = r["friedman"]; print(f"\n--- width {frac}: {scores.shape[0]} blocks | Friedman chi2={fr['chi2']:.2f} p={fr['p_chi2']:.2e} | "
                                  f"Iman-Davenport F={fr['F_iman_davenport']:.2f} p={fr['p_F']:.2e} | Nemenyi CD={r['cd']:.2f}")
        print(fr["mean_rank"].round(2).to_string())
        ranks.append(fr["mean_rank"].rename("mean_rank").reset_index().rename(columns={"index": "method"}).assign(frac=frac, chi2=fr["chi2"], p_chi2=fr["p_chi2"], cd=r["cd"]))
        pairs.append(r["pairwise"].assign(frac=frac))
        within.append(A.paired_seed_tests(sub, "block", "method", "seed", "delta", control="lsc_qr").assign(frac=frac))
    save(pd.concat(ranks), "e7_ranks"); save(pd.concat(pairs), "e7_pairs_holm"); save(pd.concat(within), "e7_within_block")
    print("\nLSC-QR vs others (Holm-adjusted Wilcoxon across blocks):")
    print(pd.concat(pairs)[["frac", "b", "wins_a", "wins_b", "median_diff", "p_adj"]].round(3).to_string(index=False))

# =============================== run ===============================
def main():
    t_all = time.time()
    for tag, fn in (("E1", exp_E1), ("E1b", exp_E1b), ("E3", exp_E3), ("E5", exp_E5), ("E6", exp_E6), ("E6b", exp_E6b), ("E2", exp_E2), ("E7", exp_E7)):
        run(tag, fn)
    LOG["total_s"] = round(time.time() - t_all, 1)
    with open(f"{OUT}/README.txt", "w") as f:
        f.write(f"annstat {A.__version__}  preset={PRESET}  device={DEV}\nconfig: {json.dumps(P, default=str)}\n"
                f"timings(s): {json.dumps(LOG['timings'])}\nerrors: {list(LOG['errors'])}\n")
    for k, v in LOG["errors"].items(): open(f"{OUT}/error_{k}.txt", "w").write(v)
    print(f"\nDONE in {LOG['total_s']:.0f}s | failed: {list(LOG['errors']) or 'none'} | results in {OUT}")

if __name__ == "__main__":
    main()
else:
    main()
