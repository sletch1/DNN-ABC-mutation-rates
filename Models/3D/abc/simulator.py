"""Two-stage-mutation Markov branching process simulator (JTB paper's Study
2 / Section 3.2; port of `../matlab/mut2stage_bMBP.m`, verified 2026-10-02
byte-identical against github.com/lruijin/ABC_mutation-rate). Mutation
probability is a step function: p(t) = p1 for t<=tau, p2 for tau<t<=tp.
Free parameters: (p1, p2, tau); `a` fixed at 1.

- mut2stage_slow : exact cell-by-cell simulation.
- mut2stage_fast : Yule-arrival-based fast simulator, extended to two stages.
- fluc_exp_2stage: J parallel cultures -> (Z_vec, X_vec).
- summary_stat   : d_bar = mean_i sqrt(X_i/Z_i), the paper's ABC statistic.

Both simulators reduce exactly to the constant-rate model when p1==p2 (or
tau<=0/tau>=tp); checked in tests/validate_simulator.py.

p(t) is evaluated at each cell's own division time -- the only convention
the real MATLAB file implements; it has no parent/offspring choice to make
(an earlier version of this file invented one, attributing it to a
commented-out branch that does not exist in the published source -- see
matlab/mut2stage_bMBP.m's header for the correction). delta (differential
mutant growth) isn't supported by the exact algorithm either, matching the
real file; mut2stage_fast and matlab/mut2stage_bMBP_rev.m both extend to
delta!=1 via a structurally different (composition-based) construction.

Note: R's rgeom counts failures before success (support {0,1,...}); numpy's
geometric counts trials to success (support {1,2,...}) -- hence `- 1` below.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Exact simulator -- port of ../matlab/mut2stage_bMBP.m
# ---------------------------------------------------------------------------
def mut2stage_slow(Z0, a, p1, p2, tau, tp, rng: np.random.Generator, delta: float = 1.0):
    """Exact cell-by-cell two-stage simulation. Returns (Z, X) at time `tp`.
    tau is the jumping time between stages ("jmpt" in MATLAB), tp is the
    checking/plating time ("chkt"). Cost grows like exp(a*tp): tp=10 gives
    ~2.2e4 cells/culture (cheap), tp=20 gives ~5e8 (not).

    `delta` must stay 1.0 -- the real MATLAB file has no differential-growth
    parameter; accepted here only so callers don't need a separate code
    path (mut2stage_fast / matlab/mut2stage_bMBP_rev.m support delta!=1 via
    a different, composition-based construction).

    Each `while` pass processes one generation as a vectorized batch: every
    surviving cell splits into 2 (`np.repeat(..., 2)`), each child becomes/
    stays mutant based on p(t) evaluated at its OWN division time, draws a
    fresh exponential division time, and exits into Z (and X if mutant)
    once past `tp`.
    """
    if not np.isclose(delta, 1.0):
        raise ValueError("mut2stage_slow has no delta support (neither does the "
                         "real exact MATLAB simulator); use mut2stage_fast for delta != 1.")

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
        parent_mut = np.repeat(mvec_last, 2)
        dtvec = np.repeat(dtvec_last, 2) + rng.exponential(1.0 / a, size=2 * n_continue)
        mu = np.where(dtvec <= tau, p1, p2)
        prob = (1 - mu) * parent_mut + mu   # mutant parent -> mutant child w.p. 1
        mvec = rng.binomial(1, prob)

        f_continue = dtvec < tp
        n_continue = int(f_continue.sum())
        Z += int((~f_continue).sum())
        X += int(((~f_continue) & (mvec == 1)).sum())

    return Z, X


# ---------------------------------------------------------------------------
# Fast simulator -- Algorithm 4 extended to two stages
# ---------------------------------------------------------------------------
def mut2stage_fast(Z0, a, p1, p2, tau, tp, rng: np.random.Generator,
                   delta: float = 1.0, stochastic_m: bool = False):
    """Fast two-stage simulator. Returns (Z, X). O(#mutants) per culture.
    The extension the paper's footnote calls "easily extended" but never
    uploaded; makes the paper's own Study 2 regime (p~1e-9, tp=20) reachable
    at all -- the exact simulator there needs ~5e8 cells/culture.

    Draws total population Z from the Yule size law, then seeds mutations at
    arrival times from the Yule arrival CDF F(t) = (exp(a*t)-1)/(exp(a*tp)-1).
    Two stages only change seeding: M1 = round(Z*p1*F(tau)) arrivals in
    [0,tau], M2 = round(Z*p2*(1-F(tau))) in (tau,tp], each clone contributing
    1 + geometric(exp(-a*delta*(tp-t_m))) cells. p1==p2 recovers the original
    (constant-rate) algorithm.

    stochastic_m=False (default, faithful to R/MATLAB) uses deterministic
    round(Z*p*F), a poor approximation when Z*p=O(1) (the paper's regime:
    Z~5e8, p~1e-9); set True to draw M1/M2 ~ Binomial instead.

    Not a port of matlab/mut2stage_bMBP_rev.m (the real fast algorithm,
    which composes two mut_bMBP_rev calls) -- this is an independent
    inverse-CDF/Yule-arrival construction, unverified against it, and not
    currently used by any deployed result (every caller in this package
    runs with use_slow=True; check before relying on this path).
    """
    Z0 = int(Z0)
    Z = int((rng.geometric(np.exp(-a * tp), size=Z0) - 1).sum())  # Yule size: sum of Z0 geometrics
    if Z <= 0:
        return Z, 0

    expm1_tp = np.expm1(a * tp)
    if expm1_tp <= 0:
        return Z, 0
    F_tau = float(np.clip(np.expm1(a * min(tau, tp)) / expm1_tp, 0.0, 1.0))
    if tau <= 0:
        F_tau = 0.0

    if stochastic_m:
        M1 = int(rng.binomial(Z, min(p1 * F_tau, 1.0)))
        M2 = int(rng.binomial(Z, min(p2 * (1.0 - F_tau), 1.0)))
    else:
        M1 = int(round(Z * p1 * F_tau))
        M2 = int(round(Z * p2 * (1.0 - F_tau)))

    X = 0
    for M, u_lo, u_hi in ((M1, 0.0, F_tau), (M2, F_tau, 1.0)):
        if M <= 0 or u_hi <= u_lo:
            continue
        u = u_lo + rng.random(M) * (u_hi - u_lo)  # inverse-CDF sampling, restricted to this stage
        arrtime = np.log1p(u * expm1_tp) / a
        clones = rng.geometric(np.exp(-(a * delta) * (tp - arrtime))) - 1  # Yule growth over remaining time
        X += int(clones.sum() + M)

    return Z, min(X, Z)


# ---------------------------------------------------------------------------
# Fluctuation experiment + summary statistic
# ---------------------------------------------------------------------------
def fluc_exp_2stage(Z0, a, p1, p2, tau, tp, J, rng: np.random.Generator,
                    use_slow=False, delta: float = 1.0, stochastic_m: bool = False):
    """J parallel cultures -> (Z_vec, X_vec), each of length J."""
    Z_vec = np.empty(J, dtype=float)
    X_vec = np.empty(J, dtype=float)
    for i in range(J):
        if use_slow:
            Z, X = mut2stage_slow(Z0, a, p1, p2, tau, tp, rng, delta=delta)
        else:
            Z, X = mut2stage_fast(Z0, a, p1, p2, tau, tp, rng,
                                  delta=delta, stochastic_m=stochastic_m)
        Z_vec[i] = Z
        X_vec[i] = X
    return Z_vec, X_vec


# The paper uses a different summary-statistic root per study: root=2
# (sqrt(X/Z)) for constant-rate (1-D, Fig. 1 of Lu/Zhu/Wu 2023); root=4
# ((X/Z)^(1/4)) for the two-stage model this module simulates (Sec. 2.2:
# keeps the response curve "smooth and non-flat" under piecewise-constant p).
SUMMARY_ROOT = 4


def summary_stat(Z_vec, X_vec, root: int = SUMMARY_ROOT) -> float:
    """d_bar = mean_i (X_i/Z_i)^(1/root); extinct cultures (Z_i=0) contribute 0."""
    Z_vec = np.asarray(Z_vec, dtype=float)
    X_vec = np.asarray(X_vec, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (X_vec / Z_vec) ** (1.0 / root)
    d[~np.isfinite(d)] = 0.0
    return float(np.mean(d))


def solve_tp(Z0, a, p, c: float = 20.0) -> float:
    """Constant-rate plating time: root of Z0*(exp(a t) - exp(a t (1-2p))) - c.
    Retained only for degenerate p1==p2 cross-checks; the two-stage study
    fixes tp (20 in the paper's Study 2, 10 here) rather than solving it."""
    from scipy.optimize import brentq
    f = lambda t: Z0 * (np.exp(a * t) - np.exp(a * t * (1 - 2 * p))) - c
    lo, hi = 1.0, 30.0
    flo, fhi = f(lo), f(hi)
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
