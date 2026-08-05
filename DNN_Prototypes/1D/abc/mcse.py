"""Monte Carlo standard errors for Table 1 (MSE of p-hat), and the GPS-ABC vs.
DNN-ABC head-to-head comparison in units of their combined MCSE.

Follows Morris, White & Crowther (2019), "Using simulation studies to
evaluate statistical methods" (Statistics in Medicine): for an MSE estimator
formed by averaging per-replicate squared errors e_r = (p_hat_r - p)^2 over
R replicates, the Monte Carlo standard error of MSE_hat = mean(e_r) is
sd(e_r) / sqrt(R), i.e. the ordinary standard error of a sample mean applied
to the squared errors themselves.

Reads results/logs/raw_replicates.csv (written by run_experiments.py, one row
per (p, J, replicate) with every method's point estimate) and writes
results/tables/mcse.md: per-cell MSE, MCSE, relative MCSE, and, for the
GPS-ABC vs. DNN-ABC comparison specifically, the MSE gap in units of its
combined MCSE (the Delta/SE column reported in the paper's Table 1).

Usage:
    python mcse.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from paths import LOG_DIR, TABLE_DIR

METHODS = ["MOM", "MLE", "ABC-MCMC", "GPS-ABC", "DNN-ABC"]


def per_cell_mcse(df):
    rows = []
    for (p, J), g in df.groupby(["p_true", "J"]):
        row = {"p": p, "J": J, "R": len(g)}
        for m in METHODS:
            e = (g[m] - p) ** 2
            mse = e.mean()
            mcse = e.std(ddof=1) / np.sqrt(len(g))
            row[f"{m}_MSE"] = mse
            row[f"{m}_MCSE"] = mcse
            row[f"{m}_relMCSE_pct"] = 100 * mcse / mse if mse > 0 else np.nan
        gap = row["GPS-ABC_MSE"] - row["DNN-ABC_MSE"]
        combined_se = np.sqrt(row["GPS-ABC_MCSE"] ** 2 + row["DNN-ABC_MCSE"] ** 2)
        row["gap_GPS_minus_DNN"] = gap
        row["combined_MCSE"] = combined_se
        row["delta_over_SE"] = gap / combined_se if combined_se > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["p", "J"]).reset_index(drop=True)


def main():
    df = pd.read_csv(LOG_DIR / "raw_replicates.csv")
    res = per_cell_mcse(df)

    lines = ["# Monte Carlo standard errors (Study I, 1-D)\n"]
    lines.append(
        "MSE_hat = mean over R replicates of (p_hat - p)^2; "
        "MCSE = sd(squared errors) / sqrt(R).\n"
    )
    lines.append(
        "delta_over_SE = (GPS-ABC MSE - DNN-ABC MSE) / combined MCSE "
        "of that gap; this is Table 1's Delta/SE column.\n"
    )
    lines.append("\n| p | J | R | " + " | ".join(f"{m} MSE (MCSE)" for m in METHODS) + " | Delta/SE |")
    lines.append("|---|---|---" + "|---" * len(METHODS) + "|---|")
    for _, r in res.iterrows():
        cells = " | ".join(
            f"{r[f'{m}_MSE']:.3e} ({r[f'{m}_MCSE']:.2e})" for m in METHODS
        )
        lines.append(
            f"| {r['p']:g} | {int(r['J'])} | {int(r['R'])} | {cells} | "
            f"{r['delta_over_SE']:+.2f} |"
        )

    allrel = pd.concat([res[f"{m}_relMCSE_pct"] for m in METHODS])
    lines.append(
        f"\nRelative MCSE across all cells/methods: "
        f"min={allrel.min():.1f}%, median={allrel.median():.1f}%, "
        f"max={allrel.max():.1f}%."
    )

    out_path = TABLE_DIR / "mcse.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print(res[["p", "J", "R", "delta_over_SE"]].to_string(index=False))


if __name__ == "__main__":
    main()
