"""Unified surrogate interface for the 3-D two-stage ABC-MCMC loop.

Every surrogate honours the same contract the sampler needs:

    predict(X) -> (mean, sd)

`X` is [log10 p1, log10 p2, tau] -- one length-3 point or an (N, 3) array -- and
the return is the predicted log10(d_bar) together with a predictive standard
deviation on that same log scale. The sampler feeds that sd straight into the
acceptance probability, exactly as GPS-ABC does with its GP variance, so a
surrogate whose uncertainty is wrong produces a wrong posterior even if its mean
is perfect.

Two backends:

- DNNSurrogate3D : the trained heteroscedastic MLP (this project's method). It
  learns from ALL ~10,000 training rows, and its forward pass is O(1) in the
  training-set size, so query cost does not grow with the data budget.

- GPSurrogate3D  : the GPS-ABC baseline -- an sklearn GaussianProcessRegressor,
  deliberately capped at a small space-filling `budget` (default 300). The cap is
  not an artificial handicap: GP fitting is O(n^3) in the number of design points
  and every prediction is O(n), which is the wall the paper itself hit (it could
  afford ~1,500 design points in its 4-D study). The head-to-head is therefore
  "DNN with all the data" vs "GP with as much as a GP can take", which is the
  real operational choice.

DERIVED FEATURE. The DNN is trained on four inputs -- the three parameters plus
log10(p_eff), the time-average of the two-stage rate (see network/model.py for
the derivation). The surrogate appends that column itself when handed raw 3-D
input, so callers never have to know about it.
"""

from __future__ import annotations

import numpy as np
import torch

from model import Standardizer, add_derived


def _as_matrix(X, ncol=3):
    """Coerce to a float32 (N, ncol) array, accepting a single point as a flat list."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != ncol and X.shape[0] == ncol:
        X = X.reshape(1, ncol)
    return X


class DNNSurrogate3D:
    """Wraps a trained heteroscedastic MLP behind `predict(X) -> (mean, sd)`.

    Args:
        model, x_scaler, y_scaler: as produced by network/train.py.
        sd_scale: the split-conformal multiplier applied to the predictive sd.
        use_derived: whether the model expects the log10(p_eff) column.
        raw_inputs: True when callers pass raw 3-column [log10 p1, log10 p2, tau]
            (the ABC sampler); False when they pass an array that already has the
            derived column appended (training/evaluation, which builds it once
            up front rather than per call).
    """

    def __init__(self, model, x_scaler: Standardizer, y_scaler: Standardizer,
                 sd_scale: float = 1.0, use_derived: bool = True,
                 raw_inputs: bool = True, tp: float = 10.0):
        self.model = model.eval()
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.sd_scale = sd_scale
        self.use_derived = use_derived
        self.raw_inputs = raw_inputs
        self.tp = tp

    @torch.no_grad()
    def predict(self, X):
        ncol = 3 if self.raw_inputs else (4 if self.use_derived else 3)
        X = _as_matrix(X, ncol)
        if self.raw_inputs and self.use_derived:
            X = add_derived(X, tp=self.tp)
        xt = self.x_scaler.transform(torch.tensor(X, dtype=torch.float32))
        mu, lv = self.model(xt)
        mean = self.y_scaler.inverse(mu).squeeze(1).numpy()
        sd = self.y_scaler.inverse_std(torch.exp(0.5 * lv).squeeze(1)).numpy() * self.sd_scale
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd


class GPSurrogate3D:
    """The GPS-ABC baseline: same contract, backed by a fitted sklearn GP."""

    def __init__(self, gpr, budget: int):
        self.gpr = gpr
        self.budget = budget          # design points the GP was actually fit on

    def predict(self, X):
        X = _as_matrix(X, 3).astype(float)
        mean, sd = self.gpr.predict(X, return_std=True)
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd


def _spacefilling_indices(X, budget, seed=0):
    """Pick ~budget rows spread across the input box (greedy farthest-point).

    Emulates the Latin-hypercube design the paper used to make a small GP viable:
    keep coverage of the (log10 p1, log10 p2, tau) box rather than a random
    subsample, which would clump. Columns are standardized first so no axis
    dominates the distance.

    `d2` holds each row's squared distance to its NEAREST already-chosen point;
    after adding a point, one elementwise minimum updates it, which is far
    cheaper than recomputing all pairwise distances each round.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if budget >= n:
        return np.arange(n)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(n))]
    d2 = ((Xs - Xs[chosen[0]]) ** 2).sum(1)
    for _ in range(budget - 1):
        nxt = int(np.argmax(d2))
        chosen.append(nxt)
        d2 = np.minimum(d2, ((Xs - Xs[nxt]) ** 2).sum(1))
    return np.array(sorted(set(chosen)))


def fit_gp_surrogate_3d(X_train, y_train, budget=300, seed: int = 0):
    """Fit the GPS-ABC baseline on a space-filling subset of the two-stage data.

    Faithful to the paper, the GP is fit on replicate-level rows so its
    WhiteKernel learns the true replicate noise -- that is where its predictive
    variance comes from in the acceptance step. An anisotropic RBF (one
    length-scale per input) lets it adapt to the very different scales of
    log10(p1), log10(p2) and tau.

    `budget=None` uses every supplied row; that is slow and is meant only for the
    surrogate-quality ablation, not for the ABC comparison.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if budget is not None and budget < len(X_train):
        idx = _spacefilling_indices(X_train, budget, seed=seed)
        X_train, y_train = X_train[idx], y_train[idx]

    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   n_restarts_optimizer=2, random_state=seed)
    gpr.fit(X_train, y_train)
    return GPSurrogate3D(gpr, budget=len(X_train))
