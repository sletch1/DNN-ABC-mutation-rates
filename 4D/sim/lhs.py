"""Latin hypercube design on a box.

The published code calls MATLAB `lhsdesign(n, d)` (default criterion 'maximin',
5 random restarts); the paper cites Liao, Yan, Xia & Luo, "A fast optimal Latin
hypercube design for Gaussian process regression modeling", IWACI 2010,
DOI 10.1109/IWACI.2010.5585160. I have not been able to read that paper's
algorithm, so `optimal_lhs` is a generic maximin LHS (stratified sample, best of
`n_restarts` by minimum pairwise distance, plus pairwise-swap hill climbing) --
the same objective, NOT a transcription of Liao et al.'s procedure.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def lhs_unit(n, d, rng):
    """MATLAB lhsdesign(n,d,'criterion','none') equivalent: one uniform draw per stratum."""
    return (np.argsort(rng.random((d, n)), axis=1).T + rng.random((n, d))) / n


def optimal_lhs(n, d, rng, n_restarts=5, n_swaps=0):
    best, best_score = None, -np.inf
    for _ in range(n_restarts):
        x = lhs_unit(n, d, rng)
        s = pdist(x).min()
        for _ in range(n_swaps):   # optional column-swap hill climb (O(n^2) per eval; off by default)
            j = rng.integers(d); i1, i2 = rng.choice(n, 2, replace=False)
            y = x.copy(); y[i1, j], y[i2, j] = x[i2, j], x[i1, j]
            s2 = pdist(y).min()
            if s2 > s:
                x, s = y, s2
        if s > best_score:
            best, best_score = x, s
    return best


def design(n, box, rng, optimal=True, **kw):
    """n points in box = [[lo,hi],...]."""
    box = np.asarray(box, float)
    u = optimal_lhs(n, len(box), rng, **kw) if optimal else lhs_unit(n, len(box), rng)
    return box[:, 0] + u * (box[:, 1] - box[:, 0])
