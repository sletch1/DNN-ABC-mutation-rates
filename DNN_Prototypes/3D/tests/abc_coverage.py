"""Frequentist coverage of the ABC 95% credible intervals.

A point estimate can be accurate while its uncertainty is miscalibrated. This test
reads results/logs/raw_replicates.csv (written by run_experiments.py, which now
records each replicate's 95% CI [cilo, cihi]) and, per method per (p, a, delta, J)
cell and overall, computes the fraction of replicates whose interval contains the
true p. A trustworthy method covers ~0.95.

This is the inference-level analog of the surrogate calibration test: it checks
that DNN-ABC's tighter intervals (Table 2) are honest, not overconfident.

Writes results/tables/abc_coverage.csv and results/logs/abc_coverage.md.
"""
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd

from paths import LOG_DIR, TABLE_DIR

warnings.filterwarnings("ignore")
METHODS = ["ABC-MCMC", "GPS-ABC", "DNN-ABC"]


def main():
    raw_path = LOG_DIR / "raw_replicates.csv"
    if not raw_path.exists():
        print("no raw_replicates.csv yet -- run run_experiments.py first")
        return
    df = pd.read_csv(raw_path)

    rows = []
    for (p, a, d, J), sub in df.groupby(["p_true", "a", "delta", "J"]):
        row = {"p": p, "a": a, "delta": d, "J": J, "n": len(sub)}
        for m in METHODS:
            lo, hi = sub.get(m + "_cilo"), sub.get(m + "_cihi")
            if lo is None or hi is None:
                row[m] = np.nan
            else:
                inside = (p >= lo.to_numpy()) & (p <= hi.to_numpy())
                row[m] = float(np.mean(inside))
        rows.append(row)
    cov = pd.DataFrame(rows)
    cov.to_csv(TABLE_DIR / "abc_coverage.csv", index=False)

    overall = {}
    for m in METHODS:
        lo, hi = df.get(m + "_cilo"), df.get(m + "_cihi")
        inside = (df["p_true"].to_numpy() >= lo.to_numpy()) & (df["p_true"].to_numpy() <= hi.to_numpy())
        overall[m] = float(np.mean(inside))

    lines = ["# ABC 95% credible-interval coverage (target 0.95)\n",
             "## Overall (all cells pooled)",
             "| method | coverage | n |", "|---|---|---|"]
    for m in METHODS:
        flag = "OK" if overall[m] >= 0.90 else ("LOW" if overall[m] < 0.90 else "")
        lines.append(f"| {m} | {overall[m]:.3f} | {len(df)} | {flag}")
    lines += ["", "## Per cell", "| p | a | delta | J | " + " | ".join(METHODS) + " |",
              "|---|---|---|---|---|---|---|"]
    for _, r in cov.iterrows():
        cells = " | ".join(f"{r[m]:.2f}" if pd.notna(r[m]) else "-" for m in METHODS)
        lines.append(f"| {r['p']:.0e} | {r['a']:.1f} | {r['delta']:.1f} | {int(r['J'])} | {cells} |")
    lines += ["",
              "Reading: coverage near 0.95 means the intervals are honest. If DNN-ABC "
              "holds ~0.95 while giving shorter intervals than GPS-ABC (Table 2), its "
              "extra precision is real, not overconfidence.", ""]
    (LOG_DIR / "abc_coverage.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\nwritten -> results/tables/abc_coverage.csv, results/logs/abc_coverage.md")


if __name__ == "__main__":
    main()
