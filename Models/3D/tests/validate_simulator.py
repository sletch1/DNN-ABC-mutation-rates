"""Validation for the two-stage simulator: checks the one structural claim
abc/simulator.py's docstring makes.

DEGENERACY: both two-stage simulators (exact and fast) reduce to the
constant-rate model when tau>=tp (every division falls in stage 1, rate p1)
or tau<=0 (every division falls in stage 2, rate p2). Checked against
Models/1D/abc/simulator.py, an independent port, so agreement is real
evidence, not a tautology. Compared in distribution (Welch t-test), not
bit-for-bit: the two ports draw their exponential/binomial random numbers in
a different order per generation, so they can't share a PRNG stream exactly
even though they sample the same distributions.

(This file previously also tested a "mut_time parent vs offspring" gap that
turned out not to exist in the published algorithm -- see
matlab/mut2stage_bMBP.m's header for the correction. The live code has
always evaluated p(t) at each cell's own division time; there was never a
second convention to measure a gap against or identify in the ground truth.)

Nothing here writes to data/ -- every check simulates fresh.

Usage: python tests/validate_simulator.py [--quick]  (full ~1min, quick ~15s/noisier)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "abc", _ROOT / "network"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
# Deliberately NOT added to sys.path: the 1-D package also contains a module named
# `simulator`, and putting its directory on the path would shadow the two-stage one
# imported just below. It is loaded by explicit file path in test_degeneracy().
_ONE_D = _ROOT.parent / "1D" / "abc"

from simulator import mut2stage_slow  # noqa: E402

_FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
    if not ok:
        _FAILURES.append(name)
    return ok


def _welch_t(a, b):
    """Welch t for two independent means; |t| < 3 is agreement at these sizes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return 0.0 if se == 0 else float((a.mean() - b.mean()) / se)


def test_degeneracy(n_sims, tp=6.0, a=1.0, p=5e-3):
    """tau >= tp must behave as constant rate p1; tau <= 0 as constant rate p2.
    Compared against the 1-D constant-rate simulator, an independent port."""
    print("\nDEGENERACY to the constant-rate model")
    # Loaded by path, not name: the 1-D module is also called `simulator` and would collide.
    import importlib.util
    spec = importlib.util.spec_from_file_location("sim1d", _ONE_D / "simulator.py")
    sim1d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim1d)

    def run_const(seed, n):
        rng = np.random.default_rng(seed)
        out = [sim1d.mut_bmbp_slow(1, a, 1.0, p, tp, rng) for _ in range(n)]
        return np.array([z for z, _ in out]), np.array([x for _, x in out])

    def run_two(seed, n, p1, p2, tau):
        rng = np.random.default_rng(seed)
        out = [mut2stage_slow(1, a, p1, p2, tau, tp, rng) for _ in range(n)]
        return np.array([z for z, _ in out]), np.array([x for _, x in out])

    OFF = 1e-12          # "off" rate for the stage that must not fire
    # p(t) is indexed by the child's own (unbounded) division time, so tau must
    # exceed any division time that can occur, not merely tp. tp+50 -> ~e^-50 escape prob.
    FAR = tp + 50.0

    for label, tau, p1, p2 in (("tau >= tp -> stage 1 only (p1)", FAR, p, OFF),
                               ("tau <= 0  -> stage 2 only (p2)", -1.0, OFF, p)):
        cZ, cX = run_const(11, 4 * n_sims)
        tZ_, tX_ = run_two(22, 4 * n_sims, p1, p2, tau)
        tX = _welch_t(cX, tX_)
        tZ = _welch_t(cZ, tZ_)
        check(label,
              abs(tX) < 3.0 and abs(tZ) < 3.0,
              f"mean X {cX.mean():8.2f} vs {tX_.mean():8.2f} (t={tX:+.2f}); "
              f"mean Z {cZ.mean():8.1f} vs {tZ_.mean():8.1f} (t={tZ:+.2f}); "
              f"n = {4*n_sims} each")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n_sims = 150 if a.quick else 600

    print("=" * 72)
    print("validate_simulator.py" + ("  [quick]" if a.quick else ""))
    print("=" * 72)
    test_degeneracy(n_sims)

    print("\n" + "=" * 72)
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}): " + "; ".join(_FAILURES))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
