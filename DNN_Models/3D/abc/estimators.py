"""Classical (non-ABC) baselines, and what they can and cannot do here.

Python port of MatlabCode/MOMMLE_fluc_exp1.m -- the paper's Eqs. (11)-(12) for
the CONSTANT-mutation-rate model.

THE HONEST CAVEAT. These estimators assume a single constant p. The two-stage
model has three parameters, and a constant-rate estimator cannot identify any of
them individually: it collapses the whole trajectory onto one number. So MOM and
MLE are not competitors to the ABC methods on (p1, p2, tau); they are included
as the naive baseline the ABC columns must beat, and as a diagnostic.

WHAT THEY ACTUALLY ESTIMATE. A mutation arising at time t founds a clone that
grows to about e^(a(tp-t)), while the number of divisions available to mutate
near t grows like e^(a*t). The exponentials cancel, so each unit of time
contributes equally and the constant rate a constant-rate estimator "sees" is
the TIME AVERAGE of the true two-stage rate:

    p_eff = ( p1 * tau + p2 * (tp - tau) ) / tp

`p_eff` below computes that target, so `estimate_mom` can be scored against the
quantity it is actually estimating rather than against p1 or p2, neither of which
it is trying to recover. This is the same quantity the surrogate takes as a
derived input feature (see network/model.py); empirically it explains R^2 = 0.874
of the design-mean variation in log10(d_bar) on its own.
"""

import warnings

import numpy as np
from scipy.optimize import fsolve


def p_eff(p1, p2, tau, tp=10.0):
    """Time-average of the two-stage mutation rate over [0, tp]."""
    tau = np.clip(tau, 0.0, tp)
    return (np.asarray(p1) * tau + np.asarray(p2) * (tp - tau)) / tp


def estimate_mom(Z_vec, X_vec) -> float:
    """Method-of-moments estimator, Eq. (12): p_hat = 0.5*(1 - log(Ybar)/log(Zbar))."""
    Z_vec = np.asarray(Z_vec, dtype=float)
    X_vec = np.asarray(X_vec, dtype=float)
    Y_bar = np.mean(Z_vec - X_vec)
    Z_bar = np.mean(Z_vec)
    if Y_bar <= 1 or Z_bar <= 1:
        return np.nan
    return (1 - np.log(Y_bar) / np.log(Z_bar)) / 2


def estimate_mle(Z_vec, X_vec) -> float:
    """MLE via the transcendental Eq. (11), solved near the MOM estimate.

    Faithful to the MATLAB `fzero(fun, st)`: a local root-find started from MOM,
    not a wide bracket, because bracketing can latch onto the spurious root near
    p = 0.5.
    """
    Z_vec = np.asarray(Z_vec, dtype=float)
    X_vec = np.asarray(X_vec, dtype=float)
    Y_bar = np.mean(Z_vec - X_vec)
    Z_bar = np.mean(Z_vec)
    if Y_bar <= 1 or Z_bar <= 1:
        return np.nan

    def fun(ph):
        return (1 - 2 * ph) * Y_bar - (1 - ph) * Z_bar ** (1 - 2 * ph) + ph

    st = max(1e-10, estimate_mom(Z_vec, X_vec))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root = float(fsolve(fun, st, xtol=1e-12)[0])
    if not np.isfinite(root) or root <= 0 or root >= 0.5:
        return estimate_mom(Z_vec, X_vec)
    return root
