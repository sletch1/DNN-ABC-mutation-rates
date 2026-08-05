"""Two-type Markov branching process (MBP) simulator for fluctuation experiments.

Direct Python port of NN_ABC/RCode/funMBP.R (the current, bug-fixed source of
truth) and the MATLAB equivalents. Constant-mutation-rate model, matching the
paper's Section 2.2 / Algorithms 2 & 4.

- mut_bmbp_slow  : Algorithm 2, exact cell-by-cell simulation (ground truth).
- mut_bmbp_fast  : Algorithm 4, fast approximate simulator (Yule/geometric shortcuts).
- fluc_exp       : J parallel cultures -> (Z_vec, X_vec).
- solve_tp       : plating time tp such that E[viable cells] hits c (=20 by default).
- summary_stat   : d_bar = mean_i sqrt(X_i / Z_i), the paper's ABC summary statistic.

R's rgeom(n, prob) counts failures before the first success (support {0,1,2,...});
numpy.random.geometric counts trials until the first success (support {1,2,...}),
so we use np.random.geometric(prob) - 1 to match R exactly.
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
    """Algorithm 2 -- exact, literal cell-by-cell simulation.

    Returns (Z, X): total viable cells and mutant cells at time tp.
    Cost grows like exp(a*tp), so this is only practical for larger p.

    Implementation note: rather than literally recursing cell-by-cell,
    each "generation" is processed as a batch over every currently-live
    lineage at once (the `dtvec`/`mvec` arrays), which is just a
    vectorized way of writing the same branching process -- every cell
    still divides independently with its own division-time and mutation
    draws, this just avoids a slow Python-level recursive tree. One pass
    of the `while` loop below = one generation:
      1. every surviving lineage splits into 2 offspring (`np.repeat(..., 2)`
         duplicates each parent's state to both children),
      2. each child is independently a mutant (prob 1 if its parent already
         was one -- mutation is irreversible here -- else prob p) via one
         binomial draw per child,
      3. each child's own division/death time is drawn from an exponential
         with rate a (non-mutants) or a*delta (mutants, who may grow at a
         different rate), added to its parent's arrival time,
      4. any lineage whose arrival time has now passed `tp` "exits" the
         simulation (it's counted into Z, and into X if it's a mutant);
         everything else continues to the next generation.
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
    """Algorithm 4 -- fast approximate simulator (Zheng 2002 shortcuts).

    Returns (Z, X). O(1)-ish per culture: draws Z from a geometric, seeds
    round(Z*p) mutations at times sampled from the truncated arrival law, and
    grows each mutant clone via a geometric.

    This replaces `mut_bmbp_slow`'s generation-by-generation simulation
    with closed-form distributional shortcuts for a pure Yule (birth-only)
    process, so the whole culture is drawn in a handful of vector ops
    instead of one exponential draw per cell per generation:
      - a Yule process started from Z0 ancestors and run to time tp has a
        known population-size distribution, so Z can be sampled directly
        (as a sum of Z0 geometrics) rather than simulated division-by-division.
      - given Z total cells, the expected mutation count is Z*p (M below);
        each mutation's arrival time is drawn from the arrival-time law
        implied by the same Yule process, restricted (truncated) to [0, tp]
        -- that's what the inverse-CDF line `arrtime = log(u*(exp(a*tp)-1)+1)/a`
        is doing, with `u` the uniform draws being inverted.
      - each mutant clone then grows independently as its own (delta-rate)
        Yule process for the remaining time `tp - arrtime`, so its final
        size is again a single geometric draw.
    This is only valid in the large-Z0/large-Z regime where the exact
    discrete branching process is well approximated by these continuous
    shortcuts -- see `mut_bmbp_slow`'s docstring for the exact alternative.
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
