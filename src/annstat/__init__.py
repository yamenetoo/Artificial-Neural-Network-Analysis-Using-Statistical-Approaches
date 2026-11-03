"""annstat - statistical tuning, inference and compression of trained neural networks."""
from . import baselines, ci, compare, input_tests, multitest, neuron_tests, plots, reduction, surrogate
from .adapters import FunctionalAdapter, ModelAdapter, SklearnMLPAdapter, from_keras, from_torch
from .ci import CI, accuracy_intervals, binomial_ci, bootstrap_interval, bootstrap_metric, clt_ci
from .input_tests import permutation_input_test
from .neuron_tests import (ablation_importance, class_separation_tests, contribution_tests,
                           redundancy_summary, weighted_outputs)
from .reduction import (ReducedModel, cluster_neurons, compress_lsc, make_cluster_model, make_pca_model,
                        reduction_path, select_medoids, select_qr)
from .surrogate import MultinomialLogit, drop_one_lr_tests, fidelity, fit_multinomial_logit, fit_ols
from .compare import compare_methods, friedman_test, nemenyi_cd, paired_seed_tests, pairwise_wilcoxon
from .tuner import StatisticalTuner, TuningReport

__version__ = "0.2.0"
__all__ = [n for n in dir() if not n.startswith("_")]
