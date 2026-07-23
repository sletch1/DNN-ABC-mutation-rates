"""Validate the Python MBP simulator port against the R/CSV ground truth.

Three checks, written to results/logs/validate_simulator.md:

1. PLATING TIME (deterministic): solve_tp(1, a, p, 20) vs the CSV's `tp` column
   at every one of the 1000 grid points. This is exact math (no randomness), so a
   correct port matches to ~1e-6. Strong pass/fail signal.

2. SUMMARY-STAT DISTRIBUTION (statistical): at a safe subset of grid points
   (p >= 1e-3, a <= 1.5 -- cheap and far from the exponential-cost / OOM regime at
   tiny p), run the Python slow simulator for many cultures and compare the mean
   log10(d_bar) to the CSV's replicate mean at that grid point. The two RNGs differ
   (numpy vs R), so this is a distributional check: agreement within Monte-Carlo
   error confirms the simulator dynamics match.

3. ESTIMATOR UNIT CHECK: MOM and MLE recover a known p on a large synthetic
   fluctuation dataset.

SAFETY: never simulates at p < 1e-3 -- those seeds need tens of GB (they are the
reps the ground-truth run itself lost to the OOM killer).
"""
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "abc", _ROOT / "figures"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd

from simulator import solve_tp, fluc_exp, summary_stat
from estimators import estimate_mom, estimate_mle
from paths import DATA, LOG_DIR

warnings.filterwarnings("ignore")


def check_tp(df):
    g = df.groupby(["p", "a"]).first().reset_index()
    err = []
    for _, r in g.iterrows():
        tp_hat = solve_tp(1.0, r["a"], r["p"], 20.0)
        err.append(abs(tp_hat - r["tp"]))
    return float(np.max(err)), float(np.mean(err)), len(g)


def check_summary(df, n_cultures=4000, seed=0):
    rng = np.random.default_rng(seed)
    sub = df[(df["p"] >= 1e-3) & (df["a"] <= 1.5)]
    keys = sub.groupby(["p", "a", "delta"]).groups.keys()
    keys = list(keys)
    rng.shuffle(keys)
    rows = []
    for (p, a, d) in keys[:8]:
        tp = solve_tp(1, a, p, 20)
        Z, X = fluc_exp(1, a, d, p, tp, n_cultures, rng, use_slow=True)
        py_mean = np.log10(max(summary_stat(Z, X), 1e-6))
        csv_mean = np.mean(np.log10(df[(np.isclose(df["p"], p)) & (np.isclose(df["a"], a))
                                       & (np.isclose(df["delta"], d))]["d_bar"]))
        rows.append((p, a, d, csv_mean, py_mean, abs(py_mean - csv_mean)))
    return rows


def check_estimators(seed=0):
    rng = np.random.default_rng(seed)
    p_true, a, delta = 5e-3, 1.0, 1.0
    tp = solve_tp(1, a, p_true, 20)
    Z, X = fluc_exp(1, a, delta, p_true, tp, 3000, rng, use_slow=True)
    return p_true, estimate_mom(Z, X), estimate_mle(Z, X)


def main():
    df = pd.read_csv(DATA)
    lines = ["# Simulator port validation\n"]

    tp_max, tp_mean, ntp = check_tp(df)
    tp_ok = tp_max < 1e-4
    lines += [f"## 1. Plating time tp (all {ntp} (p,a) grid points)",
              f"- max |tp_py - tp_csv| = {tp_max:.2e}, mean = {tp_mean:.2e}",
              f"- **{'PASS' if tp_ok else 'FAIL'}** (threshold 1e-4)\n"]

    rows = check_summary(df)
    max_d = max(r[5] for r in rows)
    sum_ok = max_d < 0.15  # within ~0.15 in log10 units (Monte-Carlo tolerance)
    lines += ["## 2. Summary-stat mean log10(d_bar): Python slow-sim vs CSV",
              "| p | a | delta | CSV mean | Python mean | abs diff |",
              "|---|---|---|---|---|---|"]
    for (p, a, d, c, py, e) in rows:
        lines.append(f"| {p:.0e} | {a:.2f} | {d:.2f} | {c:.3f} | {py:.3f} | {e:.3f} |")
    lines += [f"- max abs diff = {max_d:.3f}  -> **{'PASS' if sum_ok else 'CHECK'}** "
              f"(distributional, threshold 0.15)\n"]

    p_true, mom, mle = check_estimators()
    est_ok = abs(mom - p_true) / p_true < 0.5 and abs(mle - p_true) / p_true < 0.5
    lines += ["## 3. Estimator unit check (p_true=5e-3, 3000 cultures)",
              f"- MOM = {mom:.3e}, MLE = {mle:.3e}, truth = {p_true:.3e}",
              f"- **{'PASS' if est_ok else 'CHECK'}** (within 50% of truth)\n"]

    overall = "ALL CHECKS PASSED" if (tp_ok and sum_ok and est_ok) else "SOME CHECKS NEED REVIEW"
    lines += [f"## Overall: {overall}"]
    (LOG_DIR / "validate_simulator.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\nwritten -> results/logs/validate_simulator.md")


if __name__ == "__main__":
    main()
