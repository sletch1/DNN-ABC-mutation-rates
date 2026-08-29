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
"""

from __future__ import annotations

import numpy as np
import torch

from model import HeteroscedasticMLP, Standardizer


class DNNSurrogate:
    """Wraps a trained HeteroscedasticMLP (model.py) to satisfy the
    `predict(theta) -> (mean, sd)` contract described in the module
    docstring above -- the piece of glue that lets the ABC sampler treat
    the DNN exactly like it treats a GP, without knowing anything about
    tensors, standardization, or PyTorch internals.
    """

    def __init__(self, model: HeteroscedasticMLP, x_scaler: Standardizer,
                 y_scaler: Standardizer, sd_scale: float = 1.0):
        self.model = model.eval()  # inference mode: matters if/when Dropout or BatchNorm are enabled
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.sd_scale = sd_scale  # conformal calibration multiplier on predictive sd

    @torch.no_grad()  # no training happens here, so skip building the autograd graph
    def predict(self, theta):
        """theta: a scalar or array of log10(p) values (already on the
        natural/unstandardized scale). Returns `(mean, sd)` on that same
        scale -- a Python float pair if `theta` was scalar, else a pair of
        numpy arrays -- by running the network on the standardized input,
        then undoing the standardization on both outputs (mean and sd) so
        callers never have to think about the internal scaling.
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=np.float32))
        xt = self.x_scaler.transform(torch.tensor(theta).unsqueeze(1))
        mean_std, logvar_std = self.model(xt)
        mean = self.y_scaler.inverse(mean_std).squeeze(1).numpy()
        # std on standardized target -> original log scale via the y-scaler sigma.
        # exp(0.5*logvar) = exp(logvar)**0.5 = variance**0.5 = std; the model
        # predicts log-variance (see model.py) rather than std/variance
        # directly because it's numerically friendlier to train (always
        # positive automatically, no risk of the network trying to predict
        # a negative variance).
        sd_std = torch.exp(0.5 * logvar_std).squeeze(1)
        sd = self.y_scaler.inverse_std(sd_std).numpy() * self.sd_scale
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd

    # cost model note: forward pass is O(1) in training-set size n (target #2).


class GPSurrogate:
    """Same `predict(theta) -> (mean, sd)` contract as DNNSurrogate, but
    backed by a fitted scikit-learn GaussianProcessRegressor -- this is
    the GPS-ABC baseline the DNN is being compared against, wrapped so the
    sampler doesn't need separate code paths for the two methods."""

    def __init__(self, gpr, budget: int):
        self.gpr = gpr
        self.budget = budget  # number of training points the GP was fit on

    def predict(self, theta):
        theta = np.atleast_1d(np.asarray(theta, dtype=float)).reshape(-1, 1)
        # return_std=True: sklearn's GP regressor can report the posterior
        # predictive standard deviation directly (the analytic GP
        # uncertainty), not just the mean -- this is the GP's built-in
        # equivalent of the DNN's learned variance head.
        mean, sd = self.gpr.predict(theta, return_std=True)
        if mean.size == 1:
            return float(mean[0]), float(sd[0])
        return mean, sd


def fit_gp_surrogate(x_train, y_train, budget=None, seed: int = 0):
    """Fit the GPS-ABC baseline GP on the *raw replicate* training data.

    Faithful to the paper, which fits its GP on the replicate-level (x, z)
    samples (not averaged) so the WhiteKernel learns the true replicate noise --
    this is what supplies GPS-ABC's predictive variance in the acceptance step.
    (Averaging first collapses the noise and makes the GP overconfident, which
    breaks MCMC mixing.)

    budget=None uses all supplied points (fair head-to-head with the DNN, which
    sees the same training split). Passing an int subsamples to that many evenly
    spaced grid locations -- keeping all replicates there -- to emulate a smaller
    GP budget and its O(n^3) fitting/prediction ceiling for an ablation.
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
    # Standard signal + noise GP kernel: ConstantKernel * RBF is the usual
    # squared-exponential covariance (amplitude x smoothness), and adding
    # a WhiteKernel gives the GP a separate, learnable homoscedastic noise
    # term -- this is exactly the "one fixed noise level everywhere" the
    # DNN's heteroscedastic variance head is designed to improve on (see
    # model.py's module docstring). The (lo, hi) pairs after each
    # hyperparameter are bounds the optimizer searches within during
    # fitting, not fixed values.
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   n_restarts_optimizer=3, random_state=seed)
    gpr.fit(xs, y_train)
    return GPSurrogate(gpr, budget=n_used)
