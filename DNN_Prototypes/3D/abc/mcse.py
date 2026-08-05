"""Monte Carlo standard errors for Table 1 (MSE of p-hat), and the GPS-ABC vs.
DNN-ABC head-to-head comparison in units of their combined MCSE.

Same method as the 1-D package's abc/mcse.py (see that file's docstring for
the Morris, White & Crowther (2019) reference and formula); grouped here by
the full (p, a, delta, J) cell instead of (p, J) alone.

Reads results/logs/raw_replicates.csv and writes results/tables/mcse.md.

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
    for (p, a, delta, J), g in df.groupby(["p_true", "a", "delta", "J"]):
        row = {"p": p, "a": a, "delta": delta, "J": J, "R": len(g)}
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
    return pd.DataFrame(rows).sort_values(["p", "a", "delta", "J"]).reset_index(drop=True)


def main():
    df = pd.read_csv(LOG_DIR / "raw_replicates.csv")
    res = per_cell_mcse(df)

    lines = ["# Monte Carlo standard errors (Study II, 3-D)\n"]
    lines.append(
        "MSE_hat = mean over R replicates of (p_hat - p)^2; "
        "MCSE = sd(squared errors) / sqrt(R).\n"
    )
    lines.append(
        "delta_over_SE = (GPS-ABC MSE - DNN-ABC MSE) / combined MCSE "
        "of that gap; this is Table 1's Delta/SE column.\n"
    )
    lines.append(
        "\n| p | a | delta | J | R | "
        + " | ".join(f"{m} MSE (MCSE)" for m in ["ABC-MCMC", "GPS-ABC", "DNN-ABC"])
        + " | Delta/SE |"
    )
    lines.append("|---|---|---|---|---" + "|---" * 3 + "|---|")
    for _, r in res.iterrows():
        cells = " | ".join(
            f"{r[f'{m}_MSE']:.3e} ({r[f'{m}_MCSE']:.2e})"
            for m in ["ABC-MCMC", "GPS-ABC", "DNN-ABC"]
        )
        lines.append(
            f"| {r['p']:g} | {r['a']:g} | {r['delta']:g} | {int(r['J'])} | "
            f"{int(r['R'])} | {cells} | {r['delta_over_SE']:+.2f} |"
        )

    allrel = pd.concat(
        [res[f"{m}_relMCSE_pct"] for m in ["ABC-MCMC", "GPS-ABC", "DNN-ABC"]]
    )
    lines.append(
        f"\nRelative MCSE across all cells (three ABC methods): "
        f"min={allrel.min():.1f}%, median={allrel.median():.1f}%, "
        f"max={allrel.max():.1f}%."
    )
    lines.append(
        f"\nMax |Delta/SE| across all 12 cells: "
        f"{res['delta_over_SE'].abs().max():.2f}."
    )

    out_path = TABLE_DIR / "mcse.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print(res[["p", "a", "delta", "J", "R", "delta_over_SE"]].to_string(index=False))


if __name__ == "__main__":
    main()
