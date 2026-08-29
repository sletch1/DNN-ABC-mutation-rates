"""Monte Carlo standard errors for the 3-D two-stage result tables.

WHY THIS MATTERS HERE MORE THAN USUAL. Every number in TABLES.md is itself an
average over a finite number of simulated replicates, so it carries sampling
error. Comparing two methods without that error is how simulation studies
manufacture findings that do not replicate. On this model the risk is acute:
p1 and tau are weakly identified, so their per-replicate estimates are wildly
dispersed and a difference between methods can look large while being pure
noise.

Provides:
  mcse_mean  - standard error of a mean over replicates.
  mcse_rmse  - standard error of an RMSE, via the delta method.
  mcse_prop  - standard error of a proportion (used for interval coverage).
  annotate   - attach MCSEs to a table produced by run_experiments.aggregate and
               flag which method-pairs are separated by more than 2 MCSEs.

Usage:
    python mcse.py            # annotates results/tables/table1_recovery.csv
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "abc", _ROOT / "network"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd

from paths import LOG_DIR, TABLE_DIR


def mcse_mean(x):
    """SE of the mean of `x` over independent replicates."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def mcse_rmse(err):
    """SE of sqrt(mean(err^2)) by the delta method: sd(err^2) / (2*rmse*sqrt(n))."""
    e2 = np.asarray(err, float) ** 2
    e2 = e2[np.isfinite(e2)]
    if len(e2) < 2:
        return np.nan
    rmse = np.sqrt(e2.mean())
    return float(np.std(e2, ddof=1) / (2 * rmse * np.sqrt(len(e2)))) if rmse > 0 else np.nan


def mcse_prop(p, n):
    """SE of a proportion (interval coverage) -- binomial."""
    return float(np.sqrt(p * (1 - p) / n)) if n > 0 and np.isfinite(p) else np.nan


def annotate(raw_path=None, out_path=None):
    """Recompute per-cell metrics from the raw replicates WITH Monte Carlo SEs."""
    raw_path = Path(raw_path or LOG_DIR / "raw_replicates.csv")
    if not raw_path.exists():
        print(f"no raw replicates at {raw_path}; run abc/run_experiments.py first")
        return None
    raw = pd.read_csv(raw_path)
    methods = [m for m in ("ABC-MCMC", "GPS-ABC", "DNN-ABC") if f"{m}_p2" in raw.columns]

    rows = []
    for (p1, p2, tau, J), sub in raw.groupby(["p1_true", "p2_true", "tau_true", "J"]):
        for m in methods:
            for k, tv in (("p1", p1), ("p2", p2), ("tau", tau)):
                est = sub[f"{m}_{k}"].to_numpy(float)
                err = (np.log10(np.maximum(est, 1e-300)) - np.log10(tv)) \
                    if k in ("p1", "p2") else (est - tv)
                cov = sub[f"{m}_{k}_cov"].to_numpy(float)
                rows.append(dict(p1=p1, p2=p2, tau=tau, J=J, method=m, param=k,
                                 n=len(sub),
                                 rmse_log=float(np.sqrt(np.nanmean(err ** 2))),
                                 rmse_log_mcse=mcse_rmse(err),
                                 coverage=float(np.nanmean(cov)),
                                 coverage_mcse=mcse_prop(float(np.nanmean(cov)), len(cov))))
    df = pd.DataFrame(rows)

    lines = ["# Monte Carlo standard errors\n",
             "`rmse_log` with its MCSE. Two methods differ meaningfully only when the gap "
             "exceeds about 2 combined MCSEs; anything smaller is simulation noise, not "
             "evidence.\n",
             "| truth (p1,p2,tau) | J | param | method | rmse_log ± MCSE | coverage ± MCSE |",
             "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| ({r.p1:.0e}, {r.p2:.0e}, {r.tau:.0f}) | {int(r.J)} | {r['param']} | "
                     f"{r['method']} | {r.rmse_log:.3f} ± {r.rmse_log_mcse:.3f} | "
                     f"{r.coverage:.2f} ± {r.coverage_mcse:.2f} |")

    # explicit pairwise verdicts, so no one has to eyeball overlapping error bars
    lines.append("\n## Method comparisons (rmse_log)\n")
    for (p1, p2, tau, J, k), g in df.groupby(["p1", "p2", "tau", "J", "param"]):
        g = g.set_index("method")
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                if a in g.index and b in g.index:
                    d = g.loc[a, "rmse_log"] - g.loc[b, "rmse_log"]
                    se = np.hypot(g.loc[a, "rmse_log_mcse"], g.loc[b, "rmse_log_mcse"])
                    verdict = "TIE" if not np.isfinite(se) or abs(d) < 2 * se else \
                        (f"{b} better" if d > 0 else f"{a} better")
                    lines.append(f"- ({p1:.0e},{p2:.0e},{tau:.0f}) J={int(J)} `{k}`: "
                                 f"{a} vs {b}: delta = {d:+.3f} ± {se:.3f} -> **{verdict}**")

    out_path = Path(out_path or TABLE_DIR / "mcse.md")
    out_path.write_text("\n".join(lines) + "\n")
    df.to_csv(TABLE_DIR / "mcse.csv", index=False)
    print("\n".join(lines[:8]))
    print(f"\nwritten -> {out_path}")
    return df


if __name__ == "__main__":
    annotate()
