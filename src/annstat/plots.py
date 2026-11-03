"""Matplotlib helpers (all return the ``Axes``/``Figure`` so they can be restyled or saved)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist


def hidden_correlation_heatmap(H, ax=None):
    ax = ax or plt.subplots(figsize=(5, 4.2))[1]
    alive = np.asarray(H).std(0) > 1e-12
    corr = np.corrcoef(np.asarray(H)[:, alive], rowvar=False)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(corr.shape[0]), [str(i) for i in np.flatnonzero(alive)])
    ax.set_yticks(range(corr.shape[0]), [str(i) for i in np.flatnonzero(alive)])
    ax.set_xlabel("hidden neuron")
    ax.set_ylabel("hidden neuron")
    ax.figure.colorbar(im, ax=ax, label="Pearson r")
    return ax


def neuron_dendrogram(H, linkage="ward", ax=None):
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    H = np.asarray(H)
    alive = np.flatnonzero(H.std(0) > 1e-12)
    Z = hierarchy.linkage(pdist(H[:, alive].T), method=linkage)
    hierarchy.dendrogram(Z, labels=[str(i) for i in alive], ax=ax, color_threshold=0)
    ax.set_xlabel("hidden neuron")
    ax.set_ylabel("distance")
    return ax


def importance_bars(df, ax=None, value="loss_increase"):
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    cols = ["tab:blue" if s else "lightgray" for s in df["significant"]]
    ax.bar(df["neuron"].astype(str), df[value], color=cols)
    if {"ci_low", "ci_high"} <= set(df.columns):
        ax.errorbar(df["neuron"].astype(str), df[value], yerr=[df[value] - df["ci_low"], df["ci_high"] - df[value]],
                    fmt="none", ecolor="black", lw=0.8)
    ax.set_xlabel("hidden neuron")
    ax.set_ylabel(value.replace("_", " "))
    return ax


def reduction_curve(df, ax=None, y="metric"):
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    for m, g in df[df.method != "full"].groupby("method"):
        ax.plot(g["k"], g[y], marker="o", label=m)
    full = df[df.method == "full"][y].iloc[0]
    ax.axhline(full, color="k", ls="--", lw=0.8, label="full network")
    ax.set_xlabel("retained features k")
    ax.set_ylabel(y)
    ax.legend(fontsize=8)
    return ax
