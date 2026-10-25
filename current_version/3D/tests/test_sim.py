"""Sanity checks: fast simulators vs exact in distribution (small tp where exact is cheap),
delta=1 reduction, LHS stratification."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import mbp, lhs

rng = np.random.default_rng(0)
tp, p = 9.0, 2e-3
ex = np.array([mbp.mut_bMBP(rng, 1, 1.0, p, tp) for _ in range(3000)])
fa = np.array([mbp.mut_bMBP_rev(rng, 1, 1.0, 1.0, p, tp) for _ in range(3000)])
print("1-stage mean Z  exact/fast", ex[:, 0].mean(), fa[:, 0].mean())
print("1-stage mean X  exact/fast", ex[:, 1].mean(), fa[:, 1].mean())
p1, p2, tau = 1e-3, 5e-3, 4.0
ex = np.array([mbp.mut2stage_bMBP(rng, 1.0, p1, p2, tau, tp) for _ in range(3000)])
fa = np.array([mbp.mut2stage_rev(rng, 1, 1.0, 1.0, p1, p2, tau, tp) for _ in range(3000)])
print("2-stage mean X  exact/fast", ex[:, 1].mean(), fa[:, 1].mean())
u = lhs.lhs_unit(50, 3, rng)
assert all(sorted((u[:, j] * 50).astype(int)) == list(range(50)) for j in range(3))
print("LHS strata ok")
