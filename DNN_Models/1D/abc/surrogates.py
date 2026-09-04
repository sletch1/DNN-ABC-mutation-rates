"""Unified surrogate interface for the ABC-MCMC loop.

Every surrogate exposes the same contract the sampler needs:

    predict(theta) -> (mean, sd)

where `theta = log10(p)` (scalar or array), `mean` is the predicted
log10(d_bar), and `sd` is the predictive standard deviation on that same
log scale. The sampler injects that sd exactly the way GPS-ABC does
(`sim = Normal(mean, sd)`, MATLAB ABC_fluc_exp1_rev.m line 57-58), so the
surrogate's calibrated uncertainty flows into the acceptance probability.

Two backends:
- DNNSurrogate : the trained heteroscedastic MLP (our method). sd is the
  per-input predictive std from the variance head, optionally rescaled by a
  conformal factor so its 95% interval has valid empirical coverage.
- GPSurrogate  : sklearn GaussianProcessRegressor, deliberately trained on a
  *small* design (default 51 points, matching the paper's GP budget) to
  faithfully reproduce the GPS-ABC column and its O(n^3) training ceiling.

Both answer the same question -- "what is d_bar at this untried theta, and
how uncertain are you?" -- from different model families. The GP gets its
predictive sd from its posterior by construction; the DNN's needs the
split-conformal correction applied in train.py first. The shared interface
is what lets the sampler swap one for the other blindly, which is precisely
what this project compares.
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
        self.model = model.eval()  # inference mode: matters if/when Dropout or BatchNorm are enabled
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.sd_scale = sd_scale  # conformal calibration multiplier on predictive sd

    @torch.no_grad()  # no training happens here, so skip building the autograd graph
    def predict(self, theta):
        """theta: scalar or array of log10(p) on the raw scale. Returns
        `(mean, sd)` on that same scale, standardization undone internally.
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=np.float32))
        xt = self.x_scaler.transform(torch.tensor(theta).unsqueeze(1))
        mean_std, logvar_std = self.model(xt)
        mean = self.y_scaler.inverse(mean_std).squeeze(1).numpy()
        # exp(0.5*logvar) = sd. The model predicts log-variance rather than sd
        # so it's automatically positive and easier to train.
        sd_std = torch.exp(0.5 * logvar_std).squeeze(1)
        sd = self.y_scaler.inverse_std(sd_std).numpy() * self.sd_scale
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd

    # cost model note: forward pass is O(1) in training-set size n (target #2).


class GPSurrogate:
    """The GPS-ABC baseline: same `predict(theta) -> (mean, sd)` contract,
    backed by a fitted scikit-learn GaussianProcessRegressor."""

    def __init__(self, gpr, budget: int):
        self.gpr = gpr
        self.budget = budget  # number of training points the GP was fit on

    def predict(self, theta):
        theta = np.atleast_1d(np.asarray(theta, dtype=float)).reshape(-1, 1)
        # return_std gives the analytic GP posterior sd -- its equivalent of
        # the DNN's learned variance head.
        mean, sd = self.gpr.predict(theta, return_std=True)
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd


def fit_gp_surrogate(x_train, y_train, budget=None, seed: int = 0):
    """Fit the GPS-ABC baseline GP on the raw (unaveraged) replicate data, as
    the paper does, so the WhiteKernel learns the real replicate noise --
    averaging first would leave the GP overconfident and break MCMC mixing.

    budget=None uses every supplied point (a fair head-to-head with the DNN,
    which sees the same split); an int subsamples evenly spaced grid locations
    to emulate a smaller GP design and its O(n^3) fitting ceiling.
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
    # Signal + noise kernel: ConstantKernel * RBF is the squared-exponential
    # covariance, and WhiteKernel adds one learnable homoscedastic noise term
    # -- the single noise level the DNN's variance head improves on. The
    # (lo, hi) pairs are search bounds for the fit, not fixed values.
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   n_restarts_optimizer=3, random_state=seed)
    gpr.fit(xs, y_train)
    return GPSurrogate(gpr, budget=n_used)
