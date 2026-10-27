"""Simulator checks (run: python tests/test_sim.py). Fast and exact simulators of the original paper:
  * the number of cells Z agrees between the fast (Algorithm 4) and exact (Algorithm 2/3) simulators;
  * the fast simulator produces fewer mutants than the exact one (about 2.3-2.5x fewer at these settings): the two use the symbol p
    for different quantities (round(pZ) events versus a per-daughter probability); this is documented in the paper, not a bug;
  * our exact two-stage simulator with differential growth reduces to the published exact one at delta = 1;
  * the Latin hypercube has one point per stratum in every coordinate."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import mbp, lhs

rng = np.random.default_rng(0); n = 3000; tp = 9.0
ex = np.array([mbp.mut_bMBP(rng, 1, 1.0, 2e-3, tp) for _ in range(n)]); fa = np.array([mbp.mut_bMBP_rev(rng, 1, 1.0, 1.0, 2e-3, tp) for _ in range(n)])
assert abs(ex[:, 0].mean() / fa[:, 0].mean() - 1) < 0.08, "cell counts disagree"
r1 = ex[:, 1].mean() / fa[:, 1].mean(); assert 1.8 < r1 < 3.0, r1
p1, p2, tau = 1e-3, 5e-3, 4.0
ex2 = np.array([mbp.mut2stage_bMBP(rng, 1.0, p1, p2, tau, tp) for _ in range(n)]); fa2 = np.array([mbp.mut2stage_rev(rng, 1, 1.0, 1.0, p1, p2, tau, tp) for _ in range(n)])
r2 = ex2[:, 1].mean() / fa2[:, 1].mean(); assert 1.8 < r2 < 3.0, r2
ed = np.array([mbp.mut2stage_exact_delta(rng, 1.0, 1.0, p1, p2, tau, tp) for _ in range(n)])
assert abs(ed[:, 1].mean() / ex2[:, 1].mean() - 1) < 0.15, "exact simulator with delta=1 does not reduce to the published one"
u = lhs.lhs_unit(50, 4, rng); assert all(sorted((u[:, j] * 50).astype(int)) == list(range(50)) for j in range(4))
print(f"ok: mean mutants exact/fast = {r1:.2f} (1-stage), {r2:.2f} (2-stage); LHS strata complete")
