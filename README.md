<p align="center"><img src="assets/logo.svg" alt="annstat" width="460"></p>

# annstat — statistical tuning, inference and compression of trained neural networks

[![Open GPU experiments in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yamenetoo/Artificial-Neural-Network-Analysis-Using-Statistical-Approaches/blob/annstat-package/notebooks/annstat_gpu_experiments.ipynb) **GPU experiment suite**
[![Open quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yamenetoo/Artificial-Neural-Network-Analysis-Using-Statistical-Approaches/blob/annstat-package/notebooks/annstat_colab.ipynb) **Quickstart, tests and CIFAR-10**

`annstat` turns the two notebooks of
[Artificial-Neural-Network-Analysis-Using-Statistical-Approaches](https://github.com/yamenetoo/Artificial-Neural-Network-Analysis-Using-Statistical-Approaches)
(companion code of *Statistical tuning of artificial neural network*, Al-Mohamad, Bevrani & Haydari, arXiv:2409.16426)
into a general package. It works on **any trained feed-forward network** through a small adapter
(scikit-learn MLPs out of the box; PyTorch/Keras through `from_torch` / `from_keras`) and on
**classification and regression**, with **any number of hidden layers**.

| Original notebook step | `annstat` generalisation |
|---|---|
| Keras Iris / IDC models, hard-coded `k=10` | `SklearnMLPAdapter`, `FunctionalAdapter`, `from_torch`, `from_keras`; any width/depth/task |
| accuracy CI by CLT / bootstrap | `binomial_ci` (Wilson, Clopper–Pearson), `bootstrap_metric` (3 schemes), `clt_ci` / `bootstrap_interval(method="basic")` kept for reproducing the paper |
| Kruskal–Wallis & Mann–Whitney on `h_k·w_kj` | `contribution_tests` — paired (Friedman/Wilcoxon) or independent tests, **Holm/BH adjusted**, effect sizes |
| – | `class_separation_tests`, `ablation_importance` (paired bootstrap), `permutation_input_test`, `redundancy_summary` |
| logistic regression on hidden neurons (statsmodels) | `fit_multinomial_logit` (NumPy/SciPy Newton, Wald CIs, LR test, McFadden R², AIC/BIC), `drop_one_lr_tests`, `fit_ols` |
| clustering / PCA of neurons | `make_cluster_model`, `make_pca_model` **and** `compress_lsc` (least-squares compensated pruning, no retraining) |
| – | `StatisticalTuner(...).analyze(...)` → one report (`.to_markdown()`) |

## Install & use
```bash
pip install -e .            # numpy, scipy, pandas, scikit-learn, matplotlib
pip install -e ".[torch]"   # optional
```
```python
import annstat as A
report = A.StatisticalTuner(fitted_mlp).analyze(X_train, y_train, X_test, y_test, reduction_ks=[3, 5, 7])
print(report.to_markdown())
```
See `examples/quickstart_iris.py`. Reproduce the paper-style experiments with `experiments/e1…e5` (+ `make_tables.py`).

## Design notes (what changed w.r.t. the paper's procedures)
* **Do not use `clt_ci` on bootstrap replicates as a confidence interval for accuracy.** Its half-width is `t·s/√B`, so it shrinks with the number of replicates; empirical coverage is ≈ `2Φ(1.96/√B)−1` (≈5 % for B=1000). Use `binomial_ci` or `bootstrap_metric`.
* Output-column contributions are computed on the *same samples*; the paired tests are the default (`paired=False` reproduces the notebooks).
* All families of p-values are multiplicity-adjusted by default.
* `cluster`/`pca` reductions shrink the output head but still need all `k` neurons at inference; `compress_lsc` genuinely removes neurons (`n_neurons_needed`).
* Torch/Keras adapters are thin wrappers that were **not** executed in the development environment (not installed there); the scikit-learn path is covered by the test-suite (`pytest`).

## Author
Mohamad Yamen Al-Mohamad, Department of Statistics, University of Tabriz. MIT licence.


## Baselines and reproducibility (added)
`annstat.baselines` re-implements, for one hidden layer, ThiNet, layer-wise OBS, data-free neuron merging, CUR-leverage selection and magnitude pruning; all are available in `reduction_path` (`thinet`, `obs`, `merge`, `lsc_leverage`, `lsc_random`, `weight_norm`).
Sequential OBS ends at the same head as least-squares compensation of its kept set (tested). The PyTorch adapter was exercised on a CNN and a 3-hidden-layer MLP (`experiments/e7_torch_deep.py`); the Keras adapter has not been run. Manuscript: `paper/main.tex`.

## Colab
`notebooks/annstat_colab.ipynb` installs the package, runs the tests, a quick Iris audit, the **CIFAR-10 experiment (E9)** on a GPU, the formal significance tests (E10: Friedman, Nemenyi, Wilcoxon-Holm via `annstat.compare`), generates the LaTeX tables and zips the results. Edit `REPO_URL`/`BRANCH` in the first cell if the repository is moved.

**GPU experiment suite.** `notebooks/annstat_gpu_experiments.ipynb` (or `experiments/gpu_suite.py` with `ANNSTAT_PRESET=quick|full`) runs, on one GPU and with results saved to Google Drive after every experiment: E1/E1b (interval coverage and the exact CLT-of-replicates law), E2 (retraining-bootstrap schemes), E3 (multiplicity with correlated neurons), E5 (effective rank and ablation tests vs width and depth), E6/E6b (compression baselines on MNIST, Fashion-MNIST, CIFAR-10 and tabular data, with the error-bound check) and E7 (Friedman, Nemenyi, Wilcoxon-Holm). Finished experiments are skipped on re-run.

`annstat.compare` (v0.2.0): `friedman_test`, `nemenyi_cd`, `pairwise_wilcoxon`, `paired_seed_tests`, `compare_methods`.


## Origin
`annstat` generalises the two notebooks in this repository (`Git_IrIsModel.ipynb`, `Git_IDCModel.ipynb`, companion code of *Statistical tuning of artificial neural network*, arXiv:2409.16426), which are kept unchanged; the original README is in `docs/ORIGINAL_README.md`.
