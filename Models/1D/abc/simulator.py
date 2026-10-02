"""Two-type Markov branching process simulator, constant-mutation-rate model
(paper's Section 2.2 / Algorithms 2 & 4). Port of RCode/funMBP.R and the
MATLAB equivalents.

- mut_bmbp_slow : Algorithm 2, exact cell-by-cell simulation (ground truth).
- mut_bmbp_fast : Algorithm 4, fast approximate simulator (Yule/geometric shortcuts).
- fluc_exp      : J parallel cultures -> (Z_vec, X_vec).
- solve_tp      : plating time tp such that E[viable cells] hits c.
- summary_stat  : d_bar = mean_i sqrt(X_i/Z_i), the ABC summary statistic.

Note: R's rgeom counts failures before success (support {0,1,...}); numpy's
geometric counts trials to success (support {1,2,...}) -- hence `- 1` below.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def solve_tp(Z0: float, a: float, p: float, c: float = 20.0) -> float:
    """Plating time tp: root of Z0*(exp(a t) - exp(a t (1-2p))) - c = 0.

    Mirrors uniroot(..., c(1,30), extendInt='yes') in the R scripts.
    """
    f = lambda t: Z0 * (np.exp(a * t) - np.exp(a * t * (1 - 2 * p))) - c
    lo, hi = 1.0, 30.0
    flo, fhi = f(lo), f(hi)
    # extendInt="yes": widen the bracket until it straddles a root
    guard = 0
    while flo * fhi > 0:
        if abs(flo) < abs(fhi):
            lo -= (hi - lo)
        else:
            hi += (hi - lo)
        flo, fhi = f(lo), f(hi)
        guard += 1
        if guard > 100:
            raise RuntimeError(f"solve_tp failed to bracket (p={p}, a={a})")
    return brentq(f, lo, hi, xtol=1e-12)


def mut_bmbp_slow(Z0, a, delta, p, tp, rng: np.random.Generator):
    """Algorithm 2 -- exact cell-by-cell simulation. Returns (Z, X): total
    viable and mutant cells at time tp. Cost grows like exp(a*tp).

    Each `while` pass is one generation, batched over all live lineages: each
    splits in two, each child is mutant w.p. p (or 1 if its parent already
    was -- irreversible), division times are drawn exponentially, and
    lineages passing `tp` exit and get counted into Z (and X if mutant).
    """
    Z0 = int(Z0)
    Z = 0
    X = 0
    dtvec = rng.exponential(1.0 / a, size=Z0)
    mvec = np.zeros(Z0, dtype=int)
    f_continue = dtvec < tp
    n_continue = int(f_continue.sum())
    Z += int((~f_continue).sum())
    X += int(((~f_continue) & (mvec == 1)).sum())

    while n_continue > 0:
        dtvec_last = dtvec[f_continue]
        mvec_last = mvec[f_continue]
        # each surviving cell splits into 2 offspring
        parent_mut = np.repeat(mvec_last, 2)
        prob = (1 - p) * parent_mut + p          # mutant parent -> mutant child w.p. 1
        mvec = rng.binomial(1, prob)
        rate_vec = np.where(mvec == 1, a * delta, a)
        dtvec = np.repeat(dtvec_last, 2) + rng.exponential(1.0 / rate_vec)
        f_continue = dtvec < tp
        n_continue = int(f_continue.sum())
        Z += int((~f_continue).sum())
        X += int(((~f_continue) & (mvec == 1)).sum())

    return Z, X


def mut_bmbp_fast(Z0, a, delta, p, tp, rng: np.random.Generator):
    """Algorithm 4 -- fast approximate simulator (Zheng 2002 shortcuts), using
    closed-form Yule-process results instead of generation-by-generation
    simulation. Valid in the large-Z regime only; use `mut_bmbp_slow` for exactness.
    """
    Z0 = int(Z0)
    # Z = sum of Z0 geometrics with success prob exp(-a*tp); R rgeom support {0,1,...}
    Z = int((rng.geometric(np.exp(-a * tp), size=Z0) - 1).sum())
    M = int(round(Z * p))
    if M > 0:
        u = rng.random(M)
        arrtime = np.log(u * (np.exp(a * tp) - 1) + 1) / a
        clones = rng.geometric(np.exp(-(a * delta) * (tp - arrtime))) - 1
        X = int(clones.sum() + M)
    else:
        X = 0
    return Z, X


def fluc_exp(Z0, a, delta, p, tp, J, rng: np.random.Generator, use_slow=False):
    """J parallel cultures -> (Z_vec, X_vec) each length J."""
    sim = mut_bmbp_slow if use_slow else mut_bmbp_fast
    Z_vec = np.empty(J, dtype=float)
    X_vec = np.empty(J, dtype=float)
    for i in range(J):
        Z, X = sim(Z0, a, delta, p, tp, rng)
        Z_vec[i] = Z
        X_vec[i] = X
    return Z_vec, X_vec


def summary_stat(Z_vec, X_vec) -> float:
    """d_bar = mean_i sqrt(X_i / Z_i); extinct cultures (Z_i=0) contribute 0."""
    Z_vec = np.asarray(Z_vec, dtype=float)
    X_vec = np.asarray(X_vec, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sqrt(X_vec / Z_vec)
    d[~np.isfinite(d)] = 0.0
    return float(np.mean(d))
