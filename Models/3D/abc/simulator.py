"""Two-stage-mutation Markov branching process simulator (JTB paper's Study
2 / Section 3.2; port of `../matlab/mut2stage_bMBP.m`). Mutation probability
is a step function: p(t) = p1 for t<=tau, p2 for tau<t<=tp. Free parameters:
(p1, p2, tau); `a` fixed at 1, `delta` fixed at 1 (threaded through as an
optional arg for the paper's 4-D extension, but see mut_time below).

- mut2stage_slow : exact cell-by-cell simulation.
- mut2stage_fast : Algorithm-4-style fast simulator, extended to two stages.
- fluc_exp_2stage: J parallel cultures -> (Z_vec, X_vec).
- summary_stat   : d_bar = mean_i sqrt(X_i/Z_i), the paper's ABC statistic.

Both simulators reduce exactly to the constant-rate model when p1==p2 (or
tau<=0/tau>=tp); checked in tests/validate_simulator.py.

**mut_time convention.** A cell born at `t_birth` divides at `t_div`; which
time indexes p(t)? "parent" uses t_birth (what the paper writes, and the only
convention the fast simulator and delta!=1 can support). "offspring" uses
t_div (mut2stage_bMBP.m's live code path). Defaults to "offspring": the
shipped ground truth carries no provenance record, but simulating its design
points under both conventions shows all 60 most-discriminating points fit
"offspring" far better (mean |error| 0.0054 vs 0.0400 log10 units, t=+11.3),
matching the R generator's default and the paper's Algorithm 3. This package
previously defaulted to "parent" while the data was generated under
"offspring" -- do not revert without regenerating the data to match.

Note: R's rgeom counts failures before success (support {0,1,...}); numpy's
geometric counts trials to success (support {1,2,...}) -- hence `- 1` below.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MUT_TIME = "offspring"


# ---------------------------------------------------------------------------
# Exact simulator -- port of ../matlab/mut2stage_bMBP.m
# ---------------------------------------------------------------------------
def mut2stage_slow(Z0, a, p1, p2, tau, tp, rng: np.random.Generator,
                   delta: float = 1.0, mut_time: str = DEFAULT_MUT_TIME):
    """Exact cell-by-cell two-stage simulation. Returns (Z, X) at time `tp`.
    tau is the jumping time between stages ("jmpt" in MATLAB), tp is the
    checking/plating time ("chkt"). delta must be 1.0 when
    mut_time="offspring" (see module docstring). Cost grows like exp(a*tp):
    tp=10 gives ~2.2e4 cells/culture (cheap), tp=20 gives ~5e8 (not).

    Each `while` pass processes one generation as a vectorized batch: every
    surviving cell splits into 2 (`np.repeat(..., 2)`), each child becomes/
    stays mutant, draws a fresh exponential division time, and exits into Z
    (and X if mutant) once past `tp`.
    """
    if mut_time not in ("parent", "offspring"):
        raise ValueError(f"mut_time must be 'parent' or 'offspring', got {mut_time!r}")
    if mut_time == "offspring" and not np.isclose(delta, 1.0):
        raise ValueError(
            "mut_time='offspring' evaluates p at the offspring's own division "
            "time, which must therefore be drawn before its mutation status; "
            "delta != 1 makes that lifetime depend on the mutation status, so "
            "the two are circular. Use mut_time='parent' for delta != 1.")

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
        dtvec_last = dtvec[f_continue]      # parents' division times = children's birth times
        mvec_last = mvec[f_continue]
        parent_mut = np.repeat(mvec_last, 2)

        if mut_time == "parent":
            # p at birth time: mutation status known before lifetime is drawn, so delta can modulate it.
            birth = np.repeat(dtvec_last, 2)
            mu = np.where(birth <= tau, p1, p2)
            prob = (1 - mu) * parent_mut + mu   # mutant parent -> mutant child w.p. 1
            mvec = rng.binomial(1, prob)
            rate_vec = np.where(mvec == 1, a * delta, a)
            dtvec = birth + rng.exponential(1.0 / rate_vec)
        else:
            # p at offspring's own division time (delta unavailable here); matches mut2stage_bMBP.m's live code.
            dtvec = np.repeat(dtvec_last, 2) + rng.exponential(1.0 / a, size=2 * n_continue)
            mu = np.where(dtvec <= tau, p1, p2)
            prob = (1 - mu) * parent_mut + mu
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
    (constant-rate) algorithm. Necessarily uses the "parent"/birth-time
    convention: a clone is seeded at its arrival time.

    stochastic_m=False (default, faithful to R/MATLAB) uses deterministic
    round(Z*p*F), a poor approximation when Z*p=O(1) (the paper's regime:
    Z~5e8, p~1e-9); set True to draw M1/M2 ~ Binomial instead.
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
                    use_slow=False, delta: float = 1.0,
                    mut_time: str = DEFAULT_MUT_TIME, stochastic_m: bool = False):
    """J parallel cultures -> (Z_vec, X_vec), each of length J."""
    Z_vec = np.empty(J, dtype=float)
    X_vec = np.empty(J, dtype=float)
    for i in range(J):
        if use_slow:
            Z, X = mut2stage_slow(Z0, a, p1, p2, tau, tp, rng,
                                  delta=delta, mut_time=mut_time)
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
