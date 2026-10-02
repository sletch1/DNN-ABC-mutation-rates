"""Unified surrogate interface for the ABC-MCMC loop: every surrogate exposes
`predict(theta) -> (mean, sd)` of log10(d_bar), so the sampler can swap one
for the other blindly -- the comparison this project is built around.

- DNNSurrogate: trained heteroscedastic MLP; sd from the variance head,
  rescaled by the split-conformal factor from train.py.
- GPSurrogate: sklearn GP, trained on a small design (default 51 points,
  matching the paper's GP budget) to reproduce GPS-ABC's O(n^3) ceiling.
"""

from __future__ import annotations

import numpy as np
import torch

from model import HeteroscedasticMLP, Standardizer


class DNNSurrogate:
    """Wraps a trained HeteroscedasticMLP behind `predict(theta) -> (mean, sd)`,
    so the sampler can treat it exactly like the GP.
    """

    def __init__(self, model: HeteroscedasticMLP, x_scaler: Standardizer,
                 y_scaler: Standardizer, sd_scale: float = 1.0):
        self.model = model.eval()  # matters if Dropout/BatchNorm are enabled
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.sd_scale = sd_scale  # conformal calibration multiplier

    @torch.no_grad()
    def predict(self, theta):
        """theta: scalar or array of log10(p). Returns (mean, sd) on that
        same raw scale, standardization undone internally."""
        theta = np.atleast_1d(np.asarray(theta, dtype=np.float32))
        xt = self.x_scaler.transform(torch.tensor(theta).unsqueeze(1))
        mean_std, logvar_std = self.model(xt)
        mean = self.y_scaler.inverse(mean_std).squeeze(1).numpy()
        sd_std = torch.exp(0.5 * logvar_std).squeeze(1)  # exp(0.5*logvar) = sd
        sd = self.y_scaler.inverse_std(sd_std).numpy() * self.sd_scale
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd


class GPSurrogate:
    """The GPS-ABC baseline: same `predict(theta) -> (mean, sd)` contract,
    backed by a fitted scikit-learn GaussianProcessRegressor."""

    def __init__(self, gpr, budget: int):
        self.gpr = gpr
        self.budget = budget  # number of training points the GP was fit on

    def predict(self, theta):
        theta = np.atleast_1d(np.asarray(theta, dtype=float)).reshape(-1, 1)
        mean, sd = self.gpr.predict(theta, return_std=True)  # analytic GP posterior sd
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd


def fit_gp_surrogate(x_train, y_train, budget=None, seed: int = 0):
    """Fit the GPS-ABC baseline GP on raw (unaveraged) replicate data, as the
    paper does, so the WhiteKernel learns real replicate noise (averaging
    first would leave the GP overconfident and break MCMC mixing).
    budget=None uses every point; an int subsamples evenly spaced grid
    locations to emulate a smaller GP design and its O(n^3) ceiling.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if budget is not None:
        grid = np.unique(x_train)
        anchors = grid[np.linspace(0, len(grid) - 1, budget).round().astype(int)]
        mask = np.isin(x_train, anchors)
        x_train, y_train = x_train[mask], y_train[mask]

    n_used = len(x_train)
    xs = x_train.reshape(-1, 1)
    # ConstantKernel*RBF = squared-exponential signal; WhiteKernel adds one
    # learnable homoscedastic noise term (the DNN's variance head improves on
    # this). (lo, hi) pairs below are search bounds, not fixed values.
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   n_restarts_optimizer=3, random_state=seed)
    gpr.fit(xs, y_train)
    return GPSurrogate(gpr, budget=n_used)
