"""Unified surrogate interface for the 3-D two-stage ABC-MCMC loop.

Every surrogate honours the same contract the sampler needs:

    predict(X) -> (mean, sd)

`X` is [log10 p1, log10 p2, tau] -- one length-3 point or an (N, 3) array -- and
the return is the predicted log10(d_bar) together with a predictive standard
deviation on that same log scale. The sampler feeds that sd straight into the
acceptance probability, exactly as GPS-ABC does with its GP variance, so a
surrogate whose uncertainty is wrong produces a wrong posterior even if its mean
is perfect.

Three backends:

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

- GPSurrogate3DReference : the same baseline built to match the professor's
  ../matlab/demoGPS_fluc_exp2.m rather than to be strong -- isotropic kernel,
  raw inputs, raw target. GPSurrogate3D is better than the paper's GP on every
  one of those axes, so reporting only it overstates what the published baseline
  achieves; both are reported.

SCALES. Each surrogate carries a `scale` attribute, "log10" or "raw", naming the
scale its predictions live on; abc_mcmc.py puts the observation on that scale
before scoring. This is not bookkeeping -- the reference does ABC on the raw
statistic while we do it in log10, and conflating the two silently yields a wrong
posterior.

INPUTS. The DNN is trained on the three raw parameters (log10 p1, log10 p2,
tau) -- exactly what the sampler proposes, with nothing derived in between. The
reference GP converts back to natural rates inside its own predict().
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
    """Wraps a trained heteroscedastic MLP behind `predict(X) -> (mean, sd)`.

    Reports on the log10 scale (see `scale` below), which is what the sampler
    assumes by default.

    Args:
        model, x_scaler, y_scaler: as produced by network/train.py.
        sd_scale: the split-conformal multiplier applied to the predictive sd.
        raw_inputs: retained for call-site compatibility; inputs are always
            the 3-column [log10 p1, log10 p2, tau].
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
    """The GPS-ABC baseline: same contract, backed by a fitted sklearn GP.

    Strengthened relative to the professor's reference (anisotropic kernel,
    log-scaled inputs, learned noise); see GPSurrogate3DReference for the
    faithful version and for what the differences cost.
    """

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
    """The GPS-ABC baseline built to match the professor's `demoGPS_fluc_exp2.m`.

    `GPSurrogate3D` above is a deliberately *strengthened* GP: anisotropic RBF,
    log-scaled inputs, a learned white-noise term. The reference implementation is
    weaker on every one of those axes, and reporting only the strengthened version
    overstates what the paper's actual baseline achieves. This class is the
    reference, so the manuscript can report both.

    Three faithful differences (see ../matlab/demoGPS_fluc_exp2.m):
      - inputs are the RAW rates (p1, p2, tau), not their log10;
      - the target is the RAW statistic S = mean sqrt(X/Z), not log10(d_bar);
      - the kernel is ISOTROPIC squared-exponential (one length scale for all
        three inputs), initialised at kparams0 = [1, 1] with sigma0 = 0.02.

    SCALE. This surrogate reports on the RAW scale (`scale = "raw"`), and the
    sampler honours that -- it is not a wrapper detail, it is the reference's own
    convention. ABC_fluc_exp1.m computes `obs = mean(sqrt(X./Z))` and scores it with
    `normpdf(obs, sim, eps)`: observation, prediction and tolerance all live on the
    raw statistic. Our own pipeline works in log10 throughout, which is a further
    departure from the reference worth stating alongside the kernel differences.

    An earlier version of this class converted to log10 by the delta method so it
    could reuse the log-scale sampler unchanged. That was wrong and is recorded
    here so it is not reintroduced: the reference GP's noise is homoscedastic on
    the raw scale, so where the statistic is small the implied log-scale sd
    explodes (sd_raw / (mean_raw ln10) reached ~1e6), turning an R^2 = 0.71
    surrogate into apparent garbage. The failure was in the transform, not the GP.

    Its genuine weakness is narrower and worth reporting honestly: on the raw
    scale it reaches R^2 ~= 0.71 against ~0.99 for an otherwise identical
    anisotropic kernel. One shared length scale cannot serve inputs whose ranges
    differ by ~200x (p1, p2 span ~0.05 while tau spans ~9.8).
    """

    scale = "raw"

    def __init__(self, gpr, budget: int):
        self.gpr = gpr
        self.budget = budget

    def predict(self, X):
        """`X` arrives as (log10 p1, log10 p2, tau) -- the sampler's coordinates.

        The reference GP was fit on the RAW rates, so the first two columns are
        exponentiated back here. Doing it inside predict keeps the sampler ignorant
        of the difference and keeps the fit faithful to demoGPS_fluc_exp2.m.
        """
        X = _as_matrix(X, 3).astype(float)
        Xr = np.column_stack([10.0 ** X[:, 0], 10.0 ** X[:, 1], X[:, 2]])
        mean, sd = self.gpr.predict(Xr, return_std=True)
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


def fit_gp_surrogate_3d_reference(p1, p2, tau, S, budget=300, seed: int = 0):
    """Fit the GPS-ABC baseline exactly as `../matlab/demoGPS_fluc_exp2.m` does.

    Unlike `fit_gp_surrogate_3d`, this takes the RAW rates and the RAW statistic:

        p1, p2 : mutation probabilities on the natural scale (NOT log10)
        tau    : transition time
        S      : mean sqrt(X/Z) per row, on the natural scale (NOT log10)

    MATLAB's `fitrgp(..., 'KernelFunction', 'squaredexponential',
    'KernelParameters', [1, 1], 'Sigma', 0.02)` is an ISOTROPIC squared exponential
    -- kparams0 is (length scale, signal sd) and Sigma the noise sd -- with all
    three fitted by MLE from those starting values. The sklearn equivalent is a
    scalar-length-scale RBF times a constant, plus a white-noise term started at
    sigma0^2. `normalize_y=False` because the reference does not centre its target.

    The space-filling budget is kept from our pipeline rather than the reference's
    full 11 x 11 x 9 factorial, so that this GP and the strengthened one are given
    the same data budget and the comparison isolates the model, not the design.
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
