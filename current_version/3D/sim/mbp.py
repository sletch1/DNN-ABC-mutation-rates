"""Line-by-line NumPy ports of the published MATLAB simulators
(github.com/lruijin/ABC_mutation-rate; local copy: Original_Paper_Code/).

  mut_bMBP          exact cell-by-cell, constant rate        (mut_bMBP.m)
  mut_bMBP_rev      fast, constant rate, differential growth (mut_bMBP_rev.m)
  mut2stage_bMBP    exact, two-stage rate                    (exact simulator/countsizeBDtree2.m)
  mut2stage_rev     fast, two-stage rate, differential growth (mut2stage_bMBP_rev.m)

MATLAB conventions kept on purpose:
  geornd(p)  = number of failures before first success (support 0,1,...),
               i.e. numpy geometric(p) - 1.
  round(x)   = half away from zero (not numpy's banker's rounding).
  Z can be 0 (a lone founder can have Z=geornd=0); callers handle it as the
  original getIniX2a does (redraw) -- see draw_stat().
"""
from __future__ import annotations

import numpy as np


def geornd(rng, p, size=None):
    return rng.geometric(p, size=size) - 1


def _geo_sum_const(rng, n, p):
    """sum of n iid geornd(p) == NegativeBinomial(n, p) (failures); exact."""
    n = int(n)
    return 0 if n <= 0 else int(rng.negative_binomial(n, p))


def mut_bMBP_rev(rng, Z0, a, delta, p, tp):
    Z = _geo_sum_const(rng, Z0, np.exp(-a * tp))
    M = int(np.floor(Z * p + 0.5))
    if M > 0:
        arr = np.log(rng.uniform(0, 1, M) * (np.exp(a * tp) - 1) + 1) / a
        X = int(geornd(rng, np.exp(-(a * delta) * (tp - arr))).sum()) + M
    else:
        X = 0
    return Z, X


def mut2stage_rev(rng, Z0, a, delta, p1, p2, tau, tp):
    Z1, X1 = mut_bMBP_rev(rng, Z0, a, delta, p1, tau)
    Z, X2 = mut_bMBP_rev(rng, Z1 - X1, a, delta, p2, tp - tau)
    X = _geo_sum_const(rng, X1, np.exp(-(a * delta) * (tp - tau))) + X1 + X2
    return Z, X


def _exact(rng, a, mu_fn, tp, Z0=1):
    """Shared generation-by-generation loop of mut_bMBP / countsizeBDtree2."""
    dt = rng.exponential(1.0 / a, Z0)
    mv = np.zeros(Z0, dtype=int)
    cont = dt < tp
    Z = int((~cont).sum())
    X = int(((~cont) & (mv == 1)).sum())
    while cont.any():
        dl, ml = dt[cont], mv[cont]
        par = np.repeat(ml, 2)
        dt = np.repeat(dl, 2) + rng.exponential(1.0 / a, 2 * len(dl))
        mu = mu_fn(dt)
        mv = rng.binomial(1, (1 - mu) * par + mu)
        cont = dt < tp
        Z += int((~cont).sum())
        X += int(((~cont) & (mv == 1)).sum())
    return Z, X


def mut_bMBP(rng, Z0, a, p, tp):
    return _exact(rng, a, lambda t: np.full_like(t, p), tp, Z0)


def mut2stage_bMBP(rng, a, p1, p2, tau, tp, Z0=1):
    # countsizeBDtree2.m: mu_f = (dtvec > t_d) + 1 -> p1 if dt <= tau else p2
    return _exact(rng, a, lambda t: np.where(t > tau, p2, p1), tp, Z0)


def draw_stat(rng, theta, kind="2stage", Z0=1, a=1.0, tp=20.0, root=4, exact=False):
    """One single-culture statistic Y=(X/Z)^(1/root), redrawing while Z==0
    (getIniX2a.m). theta: 1D (log10 p,) ; 3D (log10 p1, log10 p2, tau);
    4D (log10 p1, log10 p2, tau, log10 delta) -- the paper's ordering (code: delta first)."""
    while True:
        if kind == "1d":
            p = 10 ** theta[0]
            Z, X = mut_bMBP(rng, Z0, a, p, tp) if exact else mut_bMBP_rev(rng, Z0, a, 1.0, p, tp)
        elif kind == "3d":
            Z, X = (mut2stage_bMBP(rng, a, 10 ** theta[0], 10 ** theta[1], theta[2], tp, Z0) if exact
                    else mut2stage_rev(rng, Z0, a, 1.0, 10 ** theta[0], 10 ** theta[1], theta[2], tp))
        elif kind == "4d":
            Z, X = (mut2stage_exact_delta(rng, a, 10 ** theta[3], 10 ** theta[0], 10 ** theta[1], theta[2], tp, Z0) if exact
                    else mut2stage_rev(rng, Z0, a, 10 ** theta[3], 10 ** theta[0], 10 ** theta[1], theta[2], tp))
        else:
            raise ValueError(kind)
        if Z > 0:
            return (X / Z) ** (1.0 / root)


def mut2stage_exact_delta(rng, a, delta, p1, p2, tau, tp, Z0=1):
    """EXACT cell-by-cell two-type simulator with differential growth. NOT in the published
    repo (countsizeBDtree2.m has no delta); extension of it: structure and the p(t)-at-the-
    child's-division-time convention are unchanged, but a child that is a mutant draws its
    lifetime at rate a*delta (wild type: rate a), matching the growth law the fast
    simulator's geornd(exp(-a*delta*(tp-t))) clones assume. delta=1 reproduces mut2stage_bMBP
    in distribution. Cost ~ e^{a tp} cells per culture."""
    birth = np.zeros(Z0)
    mv = np.zeros(Z0, dtype=int)
    dt = birth + rng.exponential(1.0 / a, Z0)
    cont = dt < tp
    Z = int((~cont).sum()); X = int(((~cont) & (mv == 1)).sum())
    while cont.any():
        bl, ml = dt[cont], mv[cont]
        par = np.repeat(ml, 2); b = np.repeat(bl, 2)
        wt_dt = b + rng.exponential(1.0 / a, len(b))
        mu = np.where(wt_dt > tau, p2, p1)
        m = rng.binomial(1, (1 - mu) * par + mu)
        rate = np.where(m == 1, a * delta, a)
        dt = b + rng.exponential(1.0, len(b)) / rate
        mv = m
        cont = dt < tp
        Z += int((~cont).sum()); X += int(((~cont) & (mv == 1)).sum())
    return Z, X
