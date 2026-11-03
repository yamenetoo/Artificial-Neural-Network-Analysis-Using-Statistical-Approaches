import warnings
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

import annstat as A

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def iris():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = MLPClassifier(hidden_layer_sizes=(8,), max_iter=3000, random_state=0).fit(Xtr, ytr)
    return m, Xtr, Xte, ytr, yte


@pytest.fixture(scope="module")
def binary():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = MLPClassifier(hidden_layer_sizes=(6, 4), max_iter=3000, random_state=0).fit(Xtr, ytr)
    return m, Xtr, Xte, ytr, yte


@pytest.fixture(scope="module")
def reg():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    m = MLPRegressor(hidden_layer_sizes=(8,), max_iter=5000, random_state=0).fit(Xtr, ytr)
    return m, Xtr, Xte, ytr, yte


# ---- adapters -------------------------------------------------------------------------------
def test_adapter_reproduces_sklearn(iris, binary, reg):
    for m, Xtr, Xte, *_ in (iris, binary):
        a = A.SklearnMLPAdapter(m)
        assert np.abs(a.predict_proba(Xte) - m.predict_proba(Xte)).max() < 1e-10
        assert (a.predict(Xte) == m.predict(Xte)).all()
    m, Xtr, Xte, *_ = reg
    a = A.SklearnMLPAdapter(m)
    assert np.allclose(a.predict(Xte), m.predict(Xte))


def test_multilayer_hidden_shapes(binary):
    m, Xtr, *_ = binary
    a = A.SklearnMLPAdapter(m)
    assert a.n_hidden_layers == 2
    assert a.hidden(Xtr, 0).shape[1] == 6 and a.hidden(Xtr, -1).shape[1] == 4


def test_functional_adapter(iris):
    m, Xtr, Xte, *_ = iris
    a0 = A.SklearnMLPAdapter(m)
    W, b = a0.head()
    f = A.FunctionalAdapter(lambda X: a0.hidden(X), W, b, "classification", m.classes_)
    assert (f.predict(Xte) == m.predict(Xte)).all()


# ---- multitest ------------------------------------------------------------------------------
def test_multitest_monotone_and_bounds():
    p = np.array([0.001, 0.01, 0.03, 0.04, 0.5])
    for meth in ("bonferroni", "holm", "bh", "by"):
        adj = A.multitest.adjust(p, meth)
        assert (adj >= p - 1e-12).all() and (adj <= 1).all()
    assert (A.multitest.holm(p) <= A.multitest.bonferroni(p) + 1e-12).all()
    assert (A.multitest.benjamini_hochberg(p) <= A.multitest.holm(p) + 1e-12).all()


# ---- confidence intervals -------------------------------------------------------------------
def test_binomial_ci_known_values():
    ci = A.binomial_ci(29, 30, method="clopper-pearson")
    assert 0.82 < ci.lower < 0.84 and ci.upper > 0.998
    w = A.binomial_ci(50, 100, method="wilson")
    assert abs((w.lower + w.upper) / 2 - 0.5) < 1e-9


def test_clt_width_shrinks_with_B_but_bootstrap_does_not():
    rng = np.random.default_rng(0)
    v = rng.normal(0.9, 0.03, 20000)
    assert A.clt_ci(v[:50]).width > 5 * A.clt_ci(v).width
    assert A.bootstrap_interval(v[:50]).width / A.bootstrap_interval(v).width < 1.5


def test_bootstrap_schemes(iris):
    m, _, Xte, _, yte = iris
    a = A.SklearnMLPAdapter(m)
    for scheme in ("resample_eval", "oob_retrain", "paper_retrain"):
        ci = A.bootstrap_metric(a, Xte, yte, scheme, B=10 if scheme != "resample_eval" else 200, rng=0, max_iter=50)
        assert 0 <= ci.lower <= ci.upper <= 1


# ---- neuron tests ---------------------------------------------------------------------------
def test_neuron_tests_shapes(iris):
    m, Xtr, Xte, ytr, yte = iris
    a = A.SklearnMLPAdapter(m)
    H, (W, b) = a.hidden(Xte), a.head()
    ct = A.contribution_tests(H, W)
    assert len(ct["omnibus"]) == 8 and len(ct["pairwise"]) == 8 * 3
    ct2 = A.contribution_tests(H, W, paired=False)
    assert ct2["pairwise"]["test"].iloc[0] == "mannwhitney"
    sep = A.class_separation_tests(H, yte)
    assert (sep["p_adj"] >= sep["p_value"] - 1e-12).all()
    abl = A.ablation_importance(a, Xte, yte, B=200, rng=0)
    assert len(abl) == 8 and (abl["ci_low"] <= abl["ci_high"]).all()


def test_ablation_detects_planted_signal():
    """A network that only uses neuron 0 must show significance for neuron 0 only."""
    rng = np.random.default_rng(0)
    n = 400
    H = np.column_stack([rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)])
    y = (H[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    W = np.array([[-3.0, 3.0], [0.0, 0.0], [0.0, 0.0]])
    f = A.FunctionalAdapter(lambda X: X, W, np.zeros(2), "classification", [0, 1])
    abl = A.ablation_importance(f, H, y, mode="mean", B=500, rng=1)
    assert abl["significant"].tolist() == [True, False, False]


def test_redundancy_detects_duplicates():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 2))
    H = np.column_stack([z[:, 0], z[:, 0] * 2 + 1, z[:, 1], -z[:, 1]])
    r = A.redundancy_summary(H)
    assert r["effective_rank"] < 2.3 and r["n_highly_correlated_pairs"] == 2


def test_input_permutation(iris):
    m, _, Xte, _, yte = iris
    df = A.permutation_input_test(A.SklearnMLPAdapter(m), Xte, yte, n_perm=50, rng=0)
    assert df["loss_increase"].iloc[3] > df["loss_increase"].iloc[1]


# ---- reduction ------------------------------------------------------------------------------
def test_lsc_exact_when_pruned_neurons_are_affine():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 3))
    H = np.column_stack([base, base @ [1, 2, -1] + 0.5, base @ [0.5, 0, 1] - 2])
    W = rng.normal(size=(5, 3))
    b = rng.normal(size=3)
    red = A.compress_lsc(H, W, b, [0, 1, 2], task="regression")
    assert np.allclose(red.logits(H), H @ W + b, atol=1e-6)


def test_selectors_return_k_unique(iris):
    m, Xtr, *_ = iris
    H = A.SklearnMLPAdapter(m).hidden(Xtr)
    for sel in (A.select_qr(H, 3), A.select_medoids(H, 3)):
        assert len(set(sel.tolist())) == 3


def test_reduction_path_iris(iris):
    m, Xtr, Xte, ytr, yte = iris
    df = A.reduction_path(A.SklearnMLPAdapter(m), Xtr, ytr, Xte, yte, [3, 5])
    assert set(df.method) >= {"full", "lsc_qr", "cluster_mean", "pca"}
    lsc = df[df.method == "lsc_qr"]
    assert (lsc.n_neurons_needed == lsc.k).all()
    assert (df[df.method == "pca"].n_neurons_needed == 8).all()


# ---- surrogate ------------------------------------------------------------------------------
def test_logit_matches_sklearn_on_synthetic():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    true = np.array([0.5, -1.0, 2.0])
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.3 + X @ true)))).astype(int)
    fit = A.fit_multinomial_logit(X, y, ridge=1e-9)
    lr = LogisticRegression(C=1e9, max_iter=5000).fit(X, y)
    assert np.allclose(fit.coef[1:, 0], lr.coef_[0], atol=1e-3)
    # Wald CIs cover the truth
    tab = fit.table[fit.table.term != "const"]
    assert ((tab.ci_low < true) & (true < tab.ci_high)).all()
    assert fit.converged and fit.lr_test["p_value"] < 1e-10


def test_multinomial_predicts_and_ols():
    from sklearn.datasets import make_classification

    X, y = make_classification(600, 4, n_informative=3, n_redundant=0, n_classes=3, random_state=0)
    fit = A.fit_multinomial_logit(X, y)
    assert fit.coef.shape == (5, 2) and np.mean(fit.predict(X) == y) > 0.7
    rng = np.random.default_rng(1)
    Xr = rng.normal(size=(300, 2))
    yr = 1 + 2 * Xr[:, 0] + rng.normal(size=300)
    o = A.fit_ols(Xr, yr)
    assert abs(o["table"].coef[1] - 2) < 0.3 and o["r2"] > 0.6


# ---- tuner end to end -----------------------------------------------------------------------
def test_tuner_classification_and_regression(iris, reg):
    m, Xtr, Xte, ytr, yte = iris
    rep = A.StatisticalTuner(m).analyze(Xtr, ytr, Xte, yte, n_boot=100, n_perm=20, reduction_ks=[4])
    assert "wilson" in rep.performance and rep.surrogate is not None
    assert "annstat report" in rep.to_markdown()
    m, Xtr, Xte, ytr, yte = reg
    rep = A.StatisticalTuner(m).analyze(Xtr, ytr, Xte, yte, n_boot=100, n_perm=20, reduction_ks=[4],
                                        reduction_methods=("lsc_qr", "pca"))
    assert rep.surrogate is None and rep.reduction is not None


def test_drop_one_lr_finds_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    y = (rng.random(500) < 1 / (1 + np.exp(-(2 * X[:, 0] - 1.5 * X[:, 1])))).astype(int)
    t = A.drop_one_lr_tests(X, y)
    assert t.significant.tolist() == [True, True, False, False]
