"""Unified surrogate interface for the 3-D ABC-MCMC loop.

Every surrogate exposes the same contract the sampler needs:

    predict(X) -> (mean, sd)

where `X` is the feature matrix with columns [log10(p), a, delta] (shape (N, 3)
or a single length-3 vector), `mean` is the predicted log10(d_bar), and `sd` is
the predictive standard deviation on that same log scale. The sampler injects
that sd exactly the way GPS-ABC does, so the surrogate's calibrated uncertainty
flows into the acceptance probability.

Two backends:
- DNNSurrogate3D : the trained heteroscedastic residual MLP (our method). Uses
  ALL training rows (~5000); forward pass is O(1) in n at query time.
- GPSurrogate3D  : sklearn GaussianProcessRegressor over the 3-D inputs,
  deliberately capped at a small space-filling `budget` (default 300) to reflect
  the GP's O(n^3) training / O(n) query ceiling -- the exact wall the paper hit
  (it could afford ~200 points even in 4-D via Latin-Hypercube design). This is
  the head-to-head that exercises dnn_improvement.md targets #1 and #3: the DNN
  learns the surface from 5000 rows, the GP from a few hundred.
"""

from __future__ import annotations

import numpy as np
import torch

from model import HeteroscedasticResMLP, Standardizer


def _as_matrix(X):
    """Coerce a scalar-free feature input to a float32 (N, 3) array."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    if X.shape[1] != 3 and X.shape[0] == 3:
        X = X.reshape(1, 3)
    return X


class DNNSurrogate3D:
    def __init__(self, model: HeteroscedasticResMLP, x_scaler: Standardizer,
                 y_scaler: Standardizer, sd_scale: float = 1.0):
        self.model = model.eval()
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.sd_scale = sd_scale  # conformal calibration multiplier on predictive sd

    @torch.no_grad()
    def predict(self, X):
        X = _as_matrix(X)
        xt = self.x_scaler.transform(torch.tensor(X))
        mean_std, logvar_std = self.model(xt)
        mean = self.y_scaler.inverse(mean_std).squeeze(1).numpy()
        sd_std = torch.exp(0.5 * logvar_std).squeeze(1)
        sd = self.y_scaler.inverse_std(sd_std).numpy() * self.sd_scale
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd

    # cost model note: forward pass is O(1) in training-set size n (target #2).


class GPSurrogate3D:
    def __init__(self, gpr, budget: int):
        self.gpr = gpr
        self.budget = budget  # number of training points the GP was fit on

    def predict(self, X):
        X = _as_matrix(X).astype(float)
        mean, sd = self.gpr.predict(X, return_std=True)
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd


def _spacefilling_indices(X, budget, seed=0):
    """Pick ~budget rows spread across the 3-D input box (greedy farthest-point).

    Emulates the Latin-Hypercube / space-filling design the paper used to make a
    small GP viable: keep coverage of the (log10 p, a, delta) cube rather than a
    random subsample. Standardize columns first so no axis dominates the distance.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if budget >= n:
        return np.arange(n)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(n))
    chosen = [start]
    d2 = ((Xs - Xs[start]) ** 2).sum(1)
    for _ in range(budget - 1):
        nxt = int(np.argmax(d2))
        chosen.append(nxt)
        d2 = np.minimum(d2, ((Xs - Xs[nxt]) ** 2).sum(1))
    return np.array(sorted(set(chosen)))


def fit_gp_surrogate_3d(X_train, y_train, budget=300, seed: int = 0):
    """Fit the GPS-ABC baseline GP on a space-filling subset of the 3-D data.

    Faithful to the paper, which fits its GP on replicate-level (x, z) samples so
    the WhiteKernel learns the true replicate noise (its predictive-variance source
    in the acceptance step). An anisotropic RBF (one length-scale per input) lets
    the GP adapt to the very different scales of log10(p), a and delta.

    budget caps the number of training points (default 300) to reflect the GP's
    cubic fitting cost and its per-query cost that grows with n -- the ceiling the
    DNN does not have. budget=None uses every supplied row (slow; for the
    surrogate-quality ablation only).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if budget is not None and budget < len(X_train):
        idx = _spacefilling_indices(X_train, budget, seed=seed)
        X_train, y_train = X_train[idx], y_train[idx]

    n_used = len(X_train)
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=[1.0, 1.0, 1.0],
                    length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   n_restarts_optimizer=2, random_state=seed)
    gpr.fit(X_train, y_train)
    return GPSurrogate3D(gpr, budget=n_used)
