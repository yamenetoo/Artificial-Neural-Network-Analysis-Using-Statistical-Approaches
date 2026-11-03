import numpy as np
import annstat as A
from annstat import baselines as bl
from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _setup():
    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    m = MLPClassifier((16,), max_iter=1500, random_state=0).fit(X, y)
    a = A.SklearnMLPAdapter(m)
    return m, a, a.hidden(X), y, X


def test_obs_equals_lsc_on_its_keep_set():
    m, a, H, y, X = _setup()
    W, b = a.head()
    red, keep = bl.obs_prune(H, W, b, 6, "classification", a.classes_)
    ls = A.compress_lsc(H, W, b, keep, ridge=1e-8)
    assert np.allclose(red.W, ls.W, atol=1e-5) and np.allclose(red.b, ls.b, atol=1e-5)


def test_full_ls_beats_thinet_rescaling_on_same_keep_set():
    m, a, H, y, X = _setup()
    W, b = a.head()
    keep, s = bl.select_thinet(H, W, 6)
    Z = H @ W + b
    z_thin = H[:, keep] @ (W[keep] * s[:, None]) + b
    z_ls = A.compress_lsc(H, W, b, keep).logits(H)
    assert ((Z - z_ls) ** 2).sum() <= ((Z - z_thin) ** 2).sum() + 1e-9


def test_baselines_shapes_and_path():
    m, a, H, y, X = _setup()
    df = A.reduction_path(a, X, y, X, y, [4, 8], ("lsc_qr", "obs", "thinet", "weight_norm", "lsc_leverage", "lsc_random", "merge"),
                          incoming=(m.coefs_[-2], m.intercepts_[-2]))
    assert set(df.method) >= {"obs", "thinet", "merge", "lsc_leverage"}
    assert (df.n_neurons_needed[df.method != "full"] <= 8).all()


def test_bound_pivoted_qr_residual_is_frobenius_exact():
    from scipy.linalg import qr
    m, a, H, y, X = _setup()
    W, b = a.head()
    mm = 6
    keep = A.select_qr(H, mm)
    drop = np.setdiff1d(np.arange(H.shape[1]), keep)
    red = A.compress_lsc(H, W, b, keep, ridge=0.0)
    lhs = ((H @ W + b - red.logits(H)) ** 2).sum()
    Hc = H - H.mean(0)
    _, R, _ = qr(Hc, pivoting=True, mode="economic")
    bound = np.linalg.norm(R[mm:, mm:], "fro") ** 2 * np.linalg.norm(W[drop], 2) ** 2
    assert lsc_ok(lhs, bound)


def lsc_ok(lhs, bound):
    return lhs <= bound * (1 + 1e-6)


def test_compare_module():
    import pandas as pd
    rng = np.random.default_rng(0)
    base = rng.normal(size=(10, 1))
    scores = pd.DataFrame({"good": base[:, 0] + 2.0 + 0.05 * rng.normal(size=10), "mid": base[:, 0] + 1.0 + 0.05 * rng.normal(size=10),
                           "bad": base[:, 0] + 0.05 * rng.normal(size=10)})
    r = A.compare_methods(scores)
    assert r["friedman"]["p_chi2"] < 1e-3 and r["control"] == "good"
    assert r["pairwise"].significant.all()
    assert A.nemenyi_cd(3, 10) > 0
    # null case: identical distributions -> no significant pairs
    null = pd.DataFrame(rng.normal(size=(10, 3)), columns=list("abc"))
    assert not A.pairwise_wilcoxon(null, control="a").significant.any()
    long = pd.DataFrame({"blk": np.repeat(["x", "y"], 30), "rep": np.tile(np.repeat(np.arange(10), 3), 2),
                         "m": np.tile(list("abc"), 20), "v": rng.normal(size=60)})
    assert len(A.paired_seed_tests(long, "blk", "m", "rep", "v", "a")) == 4
