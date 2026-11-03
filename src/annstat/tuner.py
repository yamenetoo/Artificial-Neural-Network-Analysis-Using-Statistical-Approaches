"""One-call statistical audit of a trained network (the package's main entry point)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import ci as _ci
from . import input_tests, neuron_tests, reduction, surrogate
from ._utils import default_metric
from .adapters import ModelAdapter, SklearnMLPAdapter


@dataclass
class TuningReport:
    task: str
    n_eval: int
    n_neurons: int
    performance: dict
    class_separation: pd.DataFrame
    ablation: pd.DataFrame
    redundancy: dict
    contribution: Optional[dict] = None
    inputs: Optional[pd.DataFrame] = None
    surrogate: Optional[object] = None
    surrogate_fidelity: Optional[dict] = None
    reduction: Optional[pd.DataFrame] = None
    suggestions: dict = field(default_factory=dict)

    # ------------------------------------------------------------------------------------------
    def to_markdown(self, digits: int = 4) -> str:
        f = lambda df: df.round(digits).to_markdown(index=False) if hasattr(df, "to_markdown") else df.round(digits).to_string(index=False)
        out = [f"# annstat report ({self.task}, n_eval={self.n_eval}, hidden neurons={self.n_neurons})", ""]
        out.append("## Performance")
        for name, ci in self.performance.items():
            out.append(f"- **{name}**: {ci.estimate:.{digits}f}  [{ci.lower:.{digits}f}, {ci.upper:.{digits}f}]")
        out += ["", "## Neuron-response association (BH-adjusted)", f(self.class_separation)]
        out += ["", "## Ablation importance (paired bootstrap, BH-adjusted)", f(self.ablation)]
        r = self.redundancy
        out += ["", "## Redundancy",
                f"- effective rank {r['effective_rank']:.2f} of {r['n_neurons']} (participation ratio {r['participation_ratio']:.2f}); "
                f"dead neurons {r['n_dead']}; highly correlated pairs {r['n_highly_correlated_pairs']}"]
        if self.contribution is not None:
            out += ["", "## Contribution tests - omnibus", f(self.contribution["omnibus"])]
        if self.inputs is not None:
            out += ["", "## Input permutation tests", f(self.inputs)]
        if self.surrogate is not None:
            s = self.surrogate.summary()
            out += ["", "## Logistic surrogate on hidden neurons",
                    f"- McFadden R2 {s['mcfadden_r2']:.3f}, AIC {s['aic']:.1f}, BIC {s['bic']:.1f}, "
                    f"LR-test p = {s['lr_test']['p_value']:.3g}; fidelity to network: {self.surrogate_fidelity}"]
        if self.reduction is not None:
            out += ["", "## Reduction path", f(self.reduction)]
        out += ["", "## Suggestions", f"- neurons whose individual ablation does not significantly raise the loss (pruning candidates): {self.suggestions.get('weak_neurons')}",
                f"- dead neurons: {self.suggestions.get('dead_neurons')}"]
        return "\n".join(out)


class StatisticalTuner:
    """Audit a fitted network.

    Parameters
    ----------
    model : a fitted scikit-learn MLP, or any :class:`~annstat.adapters.ModelAdapter`.
    """

    def __init__(self, model, alpha: float = 0.05, level: float = 0.95, seed: Optional[int] = 0):
        self.adapter: ModelAdapter = model if isinstance(model, ModelAdapter) else SklearnMLPAdapter(model)
        self.alpha, self.level, self.seed = alpha, level, seed

    def analyze(self, X_fit, y_fit, X_eval, y_eval, *, n_boot: int = 500, n_perm: int = 100,
                reduction_ks=None, reduction_methods=("lsc_qr", "lsc_importance", "cluster_mean", "pca"),
                feature_names=None, class_names=None, input_tests_on: bool = True, surrogate_on: bool = True,
                retrain_ci: bool = False, min_effect: float = 0.0) -> TuningReport:
        a = self.adapter
        rng = np.random.default_rng(self.seed)
        X_fit, X_eval = np.asarray(X_fit, float), np.asarray(X_eval, float)
        y_fit, y_eval = np.asarray(y_fit), np.asarray(y_eval)
        H_ev = a.hidden(X_eval)
        W, b = a.head()

        # performance
        perf = {}
        if a.task == "classification":
            perf.update(_ci.accuracy_intervals(a, X_eval, y_eval, self.level, n_boot, rng))
        else:
            perf["bootstrap-R2"] = _ci.bootstrap_metric(a, X_eval, y_eval, "resample_eval", n_boot, self.level, rng=rng)
        if retrain_ci:
            perf["retrain-oob"] = _ci.bootstrap_metric(a, X_eval, y_eval, "oob_retrain", min(n_boot, 100), self.level, rng=rng)

        sep = neuron_tests.class_separation_tests(H_ev, y_eval, a.task, self.alpha)
        abl = neuron_tests.ablation_importance(a, X_eval, y_eval, B=max(n_boot, 1000), alpha=self.alpha, min_effect=min_effect, rng=rng)
        red = neuron_tests.redundancy_summary(H_ev)
        contrib = None
        if a.task == "classification" and W.shape[1] >= 3:
            contrib = neuron_tests.contribution_tests(H_ev, W, alpha=self.alpha, class_names=class_names)
        inputs = (input_tests.permutation_input_test(a, X_eval, y_eval, n_perm, self.alpha, feature_names, rng)
                  if input_tests_on else None)

        sur = fid = None
        if surrogate_on and a.task == "classification":
            sur = surrogate.fit_multinomial_logit(a.hidden(X_fit), y_fit, a.classes_, level=self.level)
            fid = surrogate.fidelity(sur.predict(H_ev), a.predict(X_eval), y_eval)

        red_df = None
        if reduction_ks is not None:
            red_df = reduction.reduction_path(a, X_fit, y_fit, X_eval, y_eval, reduction_ks, reduction_methods)

        dead = np.flatnonzero(H_ev.std(0) <= 1e-12).tolist()
        # candidates for removal: individually dispensable (ablation not significant) neurons
        weak = sorted(set(np.flatnonzero(~abl["significant"].values).tolist()) - set(dead))
        return TuningReport(a.task, len(y_eval), H_ev.shape[1], perf, sep, abl, red, contrib, inputs, sur, fid, red_df,
                            {"weak_neurons": weak, "dead_neurons": dead})
