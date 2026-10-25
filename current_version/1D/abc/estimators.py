"""Classical (non-ABC) mutation-rate estimators: MOM and MLE. Port of
../matlab/MOMMLE_fluc_exp1.m, the paper's Eq. (11)-(12); fills the "MOM/MLE"
baseline column of Table 1/2. Both use only Y_bar = mean(Z-X), Z_bar = mean(Z).
"""

import numpy as np
from scipy.optimize import brentq, fsolve


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
    """MLE via the transcendental Eq. (11):
    (1-2p)*Ybar - (1-p)*Zbar^(1-2p) + p = 0, solved near the MOM estimate.
    """
    Z_vec = np.asarray(Z_vec, dtype=float)
    X_vec = np.asarray(X_vec, dtype=float)
    Y_bar = np.mean(Z_vec - X_vec)
    Z_bar = np.mean(Z_vec)
    if Y_bar <= 1 or Z_bar <= 1:
        return np.nan

    def fun(ph):
        return (1 - 2 * ph) * Y_bar - (1 - ph) * Z_bar ** (1 - 2 * ph) + ph

    # Root nearest the MOM start (matches MATLAB's fzero) -- a wide bracket
    # can latch onto a spurious second root near ph=0.5.
    st = max(1e-10, estimate_mom(Z_vec, X_vec))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            root = float(fsolve(fun, st, xtol=1e-12, full_output=False)[0])
    if not np.isfinite(root) or root <= 0 or root >= 0.5:
        return estimate_mom(Z_vec, X_vec)  # fall back to MOM if MLE failed/diverged
    return root
