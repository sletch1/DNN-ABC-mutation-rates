"""Unified surrogate interface for the 3-D two-stage ABC-MCMC loop: every
surrogate exposes `predict(X) -> (mean, sd)` of log10(d_bar) for X =
[log10 p1, log10 p2, tau]. The sampler feeds sd straight into the acceptance
probability, so a surrogate whose uncertainty is wrong gives a wrong
posterior even with a perfect mean.

Three backends:
- DNNSurrogate3D: trained heteroscedastic MLP, learns from all ~10,000 rows;
  forward pass is O(1) in training-set size.
- GPSurrogate3D: GPS-ABC baseline, capped at a small space-filling `budget`
  (default 300) -- not an artificial handicap, GP fitting is O(n^3) and this
  is the wall the paper itself hit. The real comparison: "DNN with all the
  data" vs. "GP with as much as a GP can take."
- GPSurrogate3DReference: same baseline built to match the professor's
  demoGPS_fluc_exp2.m (isotropic kernel, raw inputs/target) rather than to be
  strong -- reported alongside GPSurrogate3D so the strengthened version
  doesn't overstate what the published baseline achieves.

Each surrogate carries a `scale` attribute ("log10" or "raw"); abc_mcmc.py
puts the observation on that scale before scoring -- the reference does ABC
on the raw statistic while ours works in log10, and conflating the two
silently yields a wrong posterior.
"""

from __future__ import annotations

import numpy as np
import torch

from model import Standardizer


def _as_matrix(X, ncol=3):
    """Coerce to a float32 (N, ncol) array, accepting a single point as a flat list."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != ncol and X.shape[0] == ncol:
        X = X.reshape(1, ncol)
    return X


class DNNSurrogate3D:
    """Wraps a trained heteroscedastic MLP behind `predict(X) -> (mean, sd)`,
    on the log10 scale (the sampler's default). `sd_scale` is the split-
    conformal multiplier; `raw_inputs` kept for call-site compatibility only.
    """

    scale = "log10"

    def __init__(self, model, x_scaler: Standardizer, y_scaler: Standardizer,
                 sd_scale: float = 1.0, raw_inputs: bool = True,
                 tp: float = 10.0):
        self.model = model.eval()
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.sd_scale = sd_scale
        self.raw_inputs = raw_inputs
        self.tp = tp

    @torch.no_grad()
    def predict(self, X):
        X = _as_matrix(X, 3)
        xt = self.x_scaler.transform(torch.tensor(X, dtype=torch.float32))
        mu, lv = self.model(xt)
        mean = self.y_scaler.inverse(mu).squeeze(1).numpy()
        sd = self.y_scaler.inverse_std(torch.exp(0.5 * lv).squeeze(1)).numpy() * self.sd_scale
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd


class GPSurrogate3D:
    """GPS-ABC baseline: same contract, backed by a fitted sklearn GP.
    Strengthened relative to the professor's reference (anisotropic kernel,
    log-scaled inputs, learned noise) -- see GPSurrogate3DReference."""

    scale = "log10"

    def __init__(self, gpr, budget: int):
        self.gpr = gpr
        self.budget = budget          # design points the GP was actually fit on

    def predict(self, X):
        X = _as_matrix(X, 3).astype(float)
        mean, sd = self.gpr.predict(X, return_std=True)
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd


class GPSurrogate3DReference:
    """GPS-ABC baseline built to match `demoGPS_fluc_exp2.m` rather than to be
    strong: raw rates (p1, p2, tau) as inputs (not log10), raw statistic
    S = mean sqrt(X/Z) as target (not log10(d_bar)), and an ISOTROPIC
    squared-exponential kernel (one length scale for all three inputs,
    initialised at kparams0=[1,1], sigma0=0.02) vs. GPSurrogate3D's
    anisotropic one.

    Reports on the RAW scale (`scale = "raw"`) -- ABC_fluc_exp1.m scores
    `obs = mean(sqrt(X./Z))` against raw predictions, and the sampler honours
    that convention. (An earlier version converted to log10 by the delta
    method instead: wrong, since the reference GP's homoscedastic raw-scale
    noise makes the implied log-scale sd explode where the statistic is
    small -- do not reintroduce that.)

    Genuine weakness, reported honestly: R^2 ~= 0.71 on the raw scale vs.
    ~0.99 for an otherwise-identical anisotropic kernel -- one shared length
    scale can't serve inputs whose ranges differ by ~200x (p1/p2 span ~0.05,
    tau spans ~9.8).
    """

    scale = "raw"

    def __init__(self, gpr, budget: int):
        self.gpr = gpr
        self.budget = budget

    def predict(self, X):
        """`X` arrives as (log10 p1, log10 p2, tau), the sampler's coordinates;
        exponentiate the first two back since the reference GP was fit on raw rates."""
        X = _as_matrix(X, 3).astype(float)
        Xr = np.column_stack([10.0 ** X[:, 0], 10.0 ** X[:, 1], X[:, 2]])
        mean, sd = self.gpr.predict(Xr, return_std=True)
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd


def _spacefilling_indices(X, budget, seed=0):
    """Pick ~budget rows spread across the input box (greedy farthest-point),
    emulating the Latin-hypercube design the paper used. Columns standardized
    first so no axis dominates distance. `d2` tracks each row's squared
    distance to its nearest chosen point, updated by elementwise min -- far
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
    """Fit the GPS-ABC baseline on a space-filling subset. Fit on
    replicate-level rows (faithful to the paper) so WhiteKernel learns real
    replicate noise; anisotropic RBF adapts to log10(p1)/log10(p2)/tau's very
    different scales. budget=None uses every row -- slow, for the surrogate-
    quality ablation only, not the ABC comparison.
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


def fit_gp_surrogate_3d_reference(p1, p2, tau, S, budget=300, seed: int = 0):
    """Fit the GPS-ABC baseline as `../matlab/demoGPS_fluc_exp2.m` does: RAW
    rates (p1, p2 natural scale, not log10) and RAW statistic S=mean sqrt(X/Z)
    as target. MATLAB's isotropic squared-exponential kernel (kparams0=[1,1],
    Sigma=0.02) maps to a scalar-length-scale RBF*constant + white-noise
    term; normalize_y=False since the reference doesn't centre its target.
    Keeps our space-filling budget rather than the reference's full 11x11x9
    factorial, so this GP and the strengthened one share a data budget and
    the comparison isolates the model, not the design.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    X_train = np.column_stack([np.asarray(p1, dtype=float),
                               np.asarray(p2, dtype=float),
                               np.asarray(tau, dtype=float)])
    y_train = np.asarray(S, dtype=float)
    if budget is not None and budget < len(X_train):
        idx = _spacefilling_indices(X_train, budget, seed=seed)
        X_train, y_train = X_train[idx], y_train[idx]

    SIGMA0 = 0.02          # 'Sigma' in the reference call
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e3))   # isotropic
              + WhiteKernel(noise_level=SIGMA0 ** 2,
                            noise_level_bounds=(1e-10, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                   n_restarts_optimizer=2, random_state=seed)
    gpr.fit(X_train, y_train)
    return GPSurrogate3DReference(gpr, budget=len(X_train))
