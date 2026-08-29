"""Two-stage-mutation Markov branching process (MBP) simulator.

This is the model the JTB paper (Lu, Zhu & Wu 2023, Section 3.2 / Study 2)
actually uses for its multi-parameter study, and the model the professor's
MATLAB reference `MatlabCode/mut2stage_bMBP.m` implements. The mutation
probability is a step function of time:

    p(t) = p1  for 0 < t <= tau
           p2  for tau < t <= tp

Free parameters of the 3-D study: (p1, p2, tau). The division rate `a` is fixed
at 1 (it is never a parameter in the paper), and the mutant relative growth rate
`delta` is fixed at 1 (the professor's two-stage MATLAB has no delta argument at
all). `delta` is still threaded through the slow simulator as an optional
argument so the same code extends to the paper's 4-D (p1, p2, tau, delta) case
without a rewrite -- see `mut_time` below for the one restriction.

Superseded model: the previous 3-D package (`../3D/`) varied (p, a, delta) under
a CONSTANT mutation rate. That is a different model, not a reparameterization,
and its `a` axis was analytically non-identifiable. See ../../updates.md.

Contents
--------
- mut2stage_slow : exact, cell-by-cell simulation (port of mut2stage_bMBP.m).
- mut2stage_fast : Algorithm-4-style fast simulator, extended to two stages.
- fluc_exp_2stage: J parallel cultures -> (Z_vec, X_vec).
- summary_stat   : d_bar = mean_i sqrt(X_i / Z_i), the paper's ABC statistic.

Both simulators reduce EXACTLY to the constant-rate model when p1 == p2 (or
when tau <= 0 / tau >= tp), which `tests/validate_simulator.py` checks against
the existing constant-rate simulator.

The `mut_time` convention (IMPORTANT -- open question for the professor)
-----------------------------------------------------------------------
A cell born at time `t_birth` (its parent's division time) divides at its own
time `t_div`. Which of the two times indexes the step function p(t)?

- mut_time="parent"   : p is evaluated at `t_birth`, i.e. at the division event
  that actually creates the (possibly mutated) daughter. This is the model as
  written in the paper, it is what the *commented-out* lines 23-24 of
  mut2stage_bMBP.m do, and it is the only convention the fast simulator can
  implement (Algorithm 4 seeds a mutant clone at its arrival/birth time). It is
  also the only one that composes with `delta`: with delta != 1 a cell's
  lifetime distribution depends on whether it is a mutant, so mutation status
  must be drawn BEFORE its division time.

- mut_time="offspring": p is evaluated at `t_div`, the daughter's own future
  division time. This is what the *live* lines 25-26 of mut2stage_bMBP.m do.

We default to "parent" because it is the paper's model, it is what the fast
simulator implements, and it is required for any delta != 1. "offspring"
reproduces the professor's uploaded file bit-for-bit and is kept so the two can
be compared directly -- `tests/validate_simulator.py` quantifies the gap. This
is question Q7 in ../../updates.md; if the professor confirms "offspring" is
intended, flip DEFAULT_MUT_TIME below and regenerate.

R's rgeom(n, prob) counts failures before the first success (support {0,1,...});
numpy.random.geometric counts trials until the first success (support {1,2,...}),
so we use np.random.geometric(prob) - 1 to match R/MATLAB exactly.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MUT_TIME = "parent"


# ---------------------------------------------------------------------------
# Exact simulator -- port of MatlabCode/mut2stage_bMBP.m
# ---------------------------------------------------------------------------
def mut2stage_slow(Z0, a, p1, p2, tau, tp, rng: np.random.Generator,
                   delta: float = 1.0, mut_time: str = DEFAULT_MUT_TIME):
    """Exact cell-by-cell two-stage simulation. Returns (Z, X) at time `tp`.

    Args:
        Z0: initial population size (1 in the paper and in the MATLAB reference).
        a: rate of the exponential lifetime of a non-mutant cell.
        p1: mutation probability during stage 1 (t <= tau).
        p2: mutation probability during stage 2 (t > tau).
        tau: the jumping time separating the two stages ("jmpt" in MATLAB).
        tp: the checking/plating time ("chkt" in MATLAB).
        rng: numpy Generator.
        delta: mutant relative growth rate; mutant lifetimes are exp(a*delta).
            Must be 1.0 when mut_time="offspring" (see the module docstring).
        mut_time: "parent" or "offspring" -- which time indexes p(t).

    Cost grows like exp(a*tp), so this is only practical for modest `tp`
    (t0=10 gives ~2.2e4 cells/culture and is cheap; t0=20 gives ~5e8 and is not).

    Like the MATLAB original, one pass of the `while` loop processes an entire
    generation as a vectorized batch over every currently-live lineage: each
    surviving cell splits into 2 offspring (`np.repeat(..., 2)` duplicates the
    parent state to both children), each child independently becomes/stays a
    mutant, each child's own division time is its parent's plus a fresh
    exponential, and any child whose division time has passed `tp` exits and is
    counted into Z (and into X if it is a mutant).
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
            # p evaluated at the birth time (= the parent's division time), so
            # mutation status is known before the lifetime is drawn and `delta`
            # can modulate that lifetime. Matches the paper and the fast sim.
            birth = np.repeat(dtvec_last, 2)
            mu = np.where(birth <= tau, p1, p2)
            prob = (1 - mu) * parent_mut + mu   # mutant parent -> mutant child w.p. 1
            mvec = rng.binomial(1, prob)
            rate_vec = np.where(mvec == 1, a * delta, a)
            dtvec = birth + rng.exponential(1.0 / rate_vec)
        else:
            # p evaluated at the offspring's OWN division time, so that time must
            # be drawn first (hence delta is not available here). Bit-for-bit the
            # live lines 25-26 of mut2stage_bMBP.m.
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

    This is the extension the paper's footnote calls "easily extended" but never
    uploaded; it is what makes the paper's own Study 2 regime (p ~ 1e-9,
    tp = 20) reachable at all -- the exact simulator there needs ~5e8 cells per
    culture.

    The constant-rate Algorithm 4 draws the total population `Z` from the Yule
    size law, seeds `round(Z*p)` mutations at arrival times from the Yule
    arrival CDF

        F(t) = (exp(a*t) - 1) / (exp(a*tp) - 1),   t in [0, tp],

    and grows each mutant clone as its own Yule process over the remaining time.
    Two stages change only the seeding step: a mutation opportunity landing at
    time t carries probability p1 if t <= tau and p2 otherwise, and the fraction
    of opportunities falling in stage 1 is exactly F(tau). So

        M1 = round(Z * p1 * F(tau)),      arrival times ~ F restricted to [0, tau]
        M2 = round(Z * p2 * (1 - F(tau))), arrival times ~ F restricted to (tau, tp]

    and each clone contributes 1 + geometric(exp(-a*delta*(tp - t_m))) cells.
    Setting p1 == p2 recovers round(Z*p1*F) + round(Z*p1*(1-F)) ~= round(Z*p),
    i.e. the original algorithm.

    Note this simulator necessarily uses the "parent"/birth-time convention: a
    clone is seeded at its arrival time, which IS the mutation event's time.

    Args:
        stochastic_m: if False (default, faithful to the uploaded R/MATLAB), the
            mutation counts are the deterministic `round(Z * p * F)`. That is a
            poor approximation when Z*p = O(1) (the paper's regime: Z ~ 5e8,
            p ~ 1e-9, so Z*p ~ 0.5 rounds to 0 or 1 deterministically), so set
            True to draw M1, M2 ~ Binomial instead.
    """
    Z0 = int(Z0)
    # Yule population size at tp, started from Z0 ancestors: a sum of Z0
    # geometrics with success probability exp(-a*tp) (R's rgeom support {0,1,...}).
    Z = int((rng.geometric(np.exp(-a * tp), size=Z0) - 1).sum())
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
        # inverse-CDF sampling of F restricted to the stage's u-interval
        u = u_lo + rng.random(M) * (u_hi - u_lo)
        arrtime = np.log1p(u * expm1_tp) / a
        # each seeded mutant grows as a Yule process over the remaining time
        clones = rng.geometric(np.exp(-(a * delta) * (tp - arrtime))) - 1
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


def summary_stat(Z_vec, X_vec, root: int = 2) -> float:
    """d_bar = mean_i (X_i / Z_i)^(1/root); extinct cultures (Z_i=0) contribute 0.

    root=2 is the paper's 1-D statistic sqrt(X/Z) and is what the 08/28 meeting
    notes specify for this study. root=4 is the fourth root the paper switches to
    for its 4-D case; kept as an option so that comparison is one argument away.
    """
    Z_vec = np.asarray(Z_vec, dtype=float)
    X_vec = np.asarray(X_vec, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (X_vec / Z_vec) ** (1.0 / root)
    d[~np.isfinite(d)] = 0.0
    return float(np.mean(d))


def solve_tp(Z0, a, p, c: float = 20.0) -> float:
    """Constant-rate plating time: root of Z0*(exp(a t) - exp(a t (1-2p))) - c.

    Retained only for the degenerate p1 == p2 cross-checks against the
    constant-rate package. The two-stage study fixes tp (= 20 in the paper's
    Study 2, = 10 for the exact-simulator design here) rather than solving it;
    see ../../updates.md Q2.
    """
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
