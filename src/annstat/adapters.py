"""Backend-agnostic view of a trained feed-forward network.

All analyses in ``annstat`` work on three ingredients only:

* ``hidden(X, layer)``  - post-activation outputs of a hidden layer,
* ``head()``            - the affine map ``(W, b)`` from the *last* hidden layer to the outputs,
* ``refit(X, y)``       - (optional) retraining, warm-started from the current weights.

Adapters exist for scikit-learn ``MLPClassifier/MLPRegressor`` (fully tested), and for PyTorch /
Keras through :func:`from_torch` / :func:`from_keras` (thin wrappers around
:class:`FunctionalAdapter`; those two need the respective library installed).

Binary classifiers with a single logit are *symmetrised* into two logits ``(-z/2, +z/2)`` so that
every downstream routine can assume ``q`` output columns and a softmax link.
"""
from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

import numpy as np

from ._utils import softmax

_ACT = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "identity": lambda z: z,
}


def _symmetrize_binary(W: np.ndarray, b: np.ndarray):
    """(k,1) head of a single-logit binary net -> (k,2) head with logits (-z/2, +z/2)."""
    return np.hstack([-W / 2.0, W / 2.0]), np.concatenate([-b / 2.0, b / 2.0])


class ModelAdapter(ABC):
    task: str  # "classification" | "regression"
    classes_: Optional[np.ndarray] = None

    # ---- required -------------------------------------------------------------------------
    @abstractmethod
    def hidden(self, X, layer: int = -1) -> np.ndarray: ...

    @abstractmethod
    def head(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(W, b)`` with shapes ``(k, q)`` and ``(q,)`` for the last hidden layer."""

    @property
    @abstractmethod
    def n_hidden_layers(self) -> int: ...

    # ---- optional -------------------------------------------------------------------------
    def refit(self, X, y, warm_start: bool = True, max_iter: Optional[int] = None, rng=None):
        raise NotImplementedError(f"{type(self).__name__} does not support refitting")

    # ---- derived --------------------------------------------------------------------------
    def logits_from_hidden(self, H, W=None, b=None) -> np.ndarray:
        if W is None or b is None:
            W, b = self.head()
        return np.asarray(H) @ W + b

    def output_from_hidden(self, H, W=None, b=None) -> np.ndarray:
        """Probabilities (classification) or raw predictions (regression)."""
        z = self.logits_from_hidden(H, W, b)
        return softmax(z) if self.task == "classification" else z

    def predict_proba(self, X) -> np.ndarray:
        if self.task != "classification":
            raise AttributeError("predict_proba is only defined for classification")
        return self.output_from_hidden(self.hidden(X))

    def predict(self, X) -> np.ndarray:
        out = self.output_from_hidden(self.hidden(X))
        if self.task == "classification":
            return np.asarray(self.classes_)[np.argmax(out, axis=1)]
        return out.ravel() if out.shape[1] == 1 else out

    @property
    def n_hidden(self) -> int:
        return self.head()[0].shape[0]

    @property
    def n_outputs(self) -> int:
        return self.head()[0].shape[1]


class SklearnMLPAdapter(ModelAdapter):
    """Wraps a *fitted* ``sklearn.neural_network.MLPClassifier`` / ``MLPRegressor``."""

    def __init__(self, model):
        if not hasattr(model, "coefs_"):
            raise ValueError("the sklearn model must be fitted before it is wrapped")
        self.model = model
        self.task = "classification" if hasattr(model, "classes_") else "regression"
        self.classes_ = np.asarray(model.classes_) if self.task == "classification" else None

    @property
    def n_hidden_layers(self) -> int:
        return len(self.model.coefs_) - 1

    def _forward_hidden(self, X):
        act = _ACT[self.model.activation]
        H = np.asarray(X, dtype=float)
        outs = []
        for W, b in zip(self.model.coefs_[:-1], self.model.intercepts_[:-1]):
            H = act(H @ W + b)
            outs.append(H)
        return outs

    def hidden(self, X, layer: int = -1) -> np.ndarray:
        return self._forward_hidden(X)[layer]

    def head(self):
        W = np.array(self.model.coefs_[-1], dtype=float)
        b = np.array(self.model.intercepts_[-1], dtype=float)
        if self.task == "classification" and W.shape[1] == 1:
            return _symmetrize_binary(W, b)
        return W, b

    def refit(self, X, y, warm_start=True, max_iter=None, rng=None):
        m = copy.deepcopy(self.model)
        if warm_start:
            m.set_params(warm_start=True)
        else:
            from sklearn.base import clone

            m = clone(self.model)
        if max_iter is not None:
            m.set_params(max_iter=max_iter)
        if rng is not None and not warm_start:
            m.set_params(random_state=int(np.random.default_rng(rng).integers(2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, y)
        return SklearnMLPAdapter(m)


class FunctionalAdapter(ModelAdapter):
    """Adapter built from plain callables - the bridge for any other framework.

    Parameters
    ----------
    hidden_fn : callable ``X -> (n, k)`` array of last-hidden-layer activations.
    W, b      : weights/bias of the output layer (``(k, q)`` / ``(q,)``). A ``(k, 1)`` head of a
                binary classifier is symmetrised automatically.
    refit_fn  : optional ``(X, y, warm_start, max_iter) -> FunctionalAdapter``.
    layer_fns : optional list of callables, one per hidden layer, enabling ``layer != -1``.
    """

    def __init__(
        self,
        hidden_fn: Callable,
        W,
        b,
        task: str = "classification",
        classes: Optional[Sequence] = None,
        refit_fn: Optional[Callable] = None,
        layer_fns: Optional[Sequence[Callable]] = None,
    ):
        W = np.asarray(W, float)
        b = np.asarray(b, float).ravel()
        if task == "classification" and W.shape[1] == 1:
            W, b = _symmetrize_binary(W, b)
        self._W, self._b = W, b
        self._hidden_fn = hidden_fn
        self._layer_fns = list(layer_fns) if layer_fns is not None else None
        self.task = task
        self.classes_ = None if classes is None else np.asarray(classes)
        if task == "classification" and self.classes_ is None:
            self.classes_ = np.arange(W.shape[1])
        self._refit_fn = refit_fn

    @property
    def n_hidden_layers(self) -> int:
        return len(self._layer_fns) if self._layer_fns else 1

    def hidden(self, X, layer: int = -1) -> np.ndarray:
        if layer == -1 or self._layer_fns is None:
            return np.asarray(self._hidden_fn(X), dtype=float)
        return np.asarray(self._layer_fns[layer](X), dtype=float)

    def head(self):
        return self._W, self._b

    def refit(self, X, y, warm_start=True, max_iter=None, rng=None):
        if self._refit_fn is None:
            raise NotImplementedError("no refit_fn was supplied to this FunctionalAdapter")
        return self._refit_fn(X, y, warm_start, max_iter)


def from_torch(model, hidden_module, head_module, task="classification", classes=None,
               refit_fn=None, batch_size: int = 8192):
    """Adapter for a PyTorch model.

    ``hidden_module`` must be the module whose *output* is the post-activation last hidden layer
    (e.g. the ``nn.ReLU`` after the last hidden ``nn.Linear``); ``head_module`` is the final
    ``nn.Linear``. A forward hook captures the activations, no change to the model is needed.
    """
    import torch  # local import: optional dependency

    W = head_module.weight.detach().cpu().numpy().T
    b = head_module.bias.detach().cpu().numpy()

    def hidden_fn(X):
        store = {}
        handle = hidden_module.register_forward_hook(lambda m, i, o: store.__setitem__("h", o.detach()))
        model.eval()
        outs = []
        try:
            Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
            with torch.no_grad():
                for s in range(0, len(Xt), batch_size):
                    model(Xt[s:s + batch_size])
                    outs.append(store["h"].cpu().numpy())
        finally:
            handle.remove()
        return np.vstack(outs)

    return FunctionalAdapter(hidden_fn, W, b, task=task, classes=classes, refit_fn=refit_fn)


def from_keras(model, hidden_layer, task="classification", classes=None, refit_fn=None):
    """Adapter for a Keras model; ``hidden_layer`` is a layer name or index (last hidden layer)."""
    import tensorflow as tf  # local import: optional dependency

    layer = model.get_layer(hidden_layer) if isinstance(hidden_layer, str) else model.layers[hidden_layer]
    sub = tf.keras.Model(inputs=model.input, outputs=layer.output)
    W, b = model.layers[-1].get_weights()

    def hidden_fn(X):
        return sub.predict(np.asarray(X, dtype="float32"), verbose=0)

    return FunctionalAdapter(hidden_fn, W, b, task=task, classes=classes, refit_fn=refit_fn)
