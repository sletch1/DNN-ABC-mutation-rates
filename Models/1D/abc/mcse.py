"""Monte Carlo standard errors for Table 1 (MSE of p-hat), and NPE's
head-to-head comparison against each of GPS-ABC and DNN-ABC in units of
their combined MCSE.

Follows Morris, White & Crowther (2019), "Using simulation studies to
evaluate statistical methods" (Statistics in Medicine): for an MSE estimator
formed by averaging per-replicate squared errors e_r = (p_hat_r - p)^2 over
R replicates, the Monte Carlo standard error of MSE_hat = mean(e_r) is
sd(e_r) / sqrt(R), i.e. the ordinary standard error of a sample mean applied
to the squared errors themselves.

Reads results/logs/raw_replicates.csv (written by run_experiments.py, one row
per (p, J, replicate) with every method's point estimate) and writes
results/tables/mcse.md: per-cell MSE, MCSE, relative MCSE, and two head-to-
head gaps in units of their combined MCSE -- GPS-ABC vs. DNN-ABC (Table 1's
Delta/SE column) and NPE vs. DNN-ABC (the M1 comparison the Introduction and
"Related work" motivate: does the amortized baseline actually beat the
surrogate-in-ABC-MCMC design this paper is about?).

Usage:
    python mcse.py

This answers: "Table 1 shows one method ahead in a given cell -- real effect,
or replicate noise?" Each cell's MSE_hat is itself a sample mean, so it
carries an ordinary standard error. `delta_over_SE` is the gap over the
standard error of that gap; below about 2 in magnitude means unresolved at
this replicate count. It is a descriptive diagnostic, not a hypothesis test.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from paths import LOG_DIR, TABLE_DIR

METHODS = ["MOM", "MLE", "ABC-MCMC", "GPS-ABC", "DNN-ABC", "NPE"]


def _gap(row, a, b):
    """(MSE_a - MSE_b) / combined MCSE, or NaN if either method is absent
    (e.g. an older raw_replicates.csv with no NPE column)."""
    if f"{a}_MSE" not in row or f"{b}_MSE" not in row:
        return np.nan, np.nan
    gap = row[f"{a}_MSE"] - row[f"{b}_MSE"]
    se = np.sqrt(row[f"{a}_MCSE"] ** 2 + row[f"{b}_MCSE"] ** 2)
    return gap, (gap / se if se > 0 else np.nan)


def per_cell_mcse(df):
    rows = []
    for (p, J), g in df.groupby(["p_true", "J"]):
        row = {"p": p, "J": J, "R": len(g)}
        for m in METHODS:
            if m not in g:
                continue  # older raw_replicates.csv without this method
            e = (g[m] - p) ** 2
            mse = e.mean()
            mcse = e.std(ddof=1) / np.sqrt(len(g))
            row[f"{m}_MSE"] = mse
            row[f"{m}_MCSE"] = mcse
            row[f"{m}_relMCSE_pct"] = 100 * mcse / mse if mse > 0 else np.nan
        row["gap_GPS_minus_DNN"], row["delta_over_SE"] = _gap(row, "GPS-ABC", "DNN-ABC")
        row["gap_NPE_minus_DNN"], row["delta_over_SE_NPE_DNN"] = _gap(row, "NPE", "DNN-ABC")
        row["gap_NPE_minus_GPS"], row["delta_over_SE_NPE_GPS"] = _gap(row, "NPE", "GPS-ABC")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["p", "J"]).reset_index(drop=True)


def main():
    df = pd.read_csv(LOG_DIR / "raw_replicates.csv")
    present = [m for m in METHODS if m in df.columns]
    res = per_cell_mcse(df)

    lines = ["# Monte Carlo standard errors (Study I, 1-D)\n"]
    lines.append(
        "MSE_hat = mean over R replicates of (p_hat - p)^2; "
        "MCSE = sd(squared errors) / sqrt(R).\n"
    )
    lines.append(
        "Delta/SE = (GPS-ABC MSE - DNN-ABC MSE) / combined MCSE of that gap "
        "(Table 1's Delta/SE column). Delta/SE(NPE-DNN) and Delta/SE(NPE-GPS) "
        "are the same construction for NPE against each surrogate.\n"
    )
    lines.append("\n| p | J | R | " + " | ".join(f"{m} MSE (MCSE)" for m in present)
                  + " | Delta/SE | Delta/SE(NPE-DNN) | Delta/SE(NPE-GPS) |")
    lines.append("|---|---|---" + "|---" * len(present) + "|---|---|---|")
    for _, r in res.iterrows():
        cells = " | ".join(
            f"{r[f'{m}_MSE']:.3e} ({r[f'{m}_MCSE']:.2e})" for m in present
        )
        def fmt(v):
            return f"{v:+.2f}" if pd.notna(v) else "-"
        lines.append(
            f"| {r['p']:g} | {int(r['J'])} | {int(r['R'])} | {cells} | "
            f"{fmt(r['delta_over_SE'])} | {fmt(r['delta_over_SE_NPE_DNN'])} | "
            f"{fmt(r['delta_over_SE_NPE_GPS'])} |"
        )

    allrel = pd.concat([res[f"{m}_relMCSE_pct"] for m in present])
    lines.append(
        f"\nRelative MCSE across all cells/methods: "
        f"min={allrel.min():.1f}%, median={allrel.median():.1f}%, "
        f"max={allrel.max():.1f}%."
    )

    out_path = TABLE_DIR / "mcse.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print(res[["p", "J", "R", "delta_over_SE", "delta_over_SE_NPE_DNN",
               "delta_over_SE_NPE_GPS"]].to_string(index=False))


if __name__ == "__main__":
    main()
