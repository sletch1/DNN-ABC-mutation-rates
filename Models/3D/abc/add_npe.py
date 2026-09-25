"""Add the NPE baseline to an already-completed 3-D accuracy sweep, without
rerunning GPS-ABC / GPS-ABC-ref / DNN-ABC (and, if present, exact ABC-MCMC).

Same rationale as the 1-D study's add_npe.py: every row in
`results/logs/raw_replicates.csv` already stores `obs`, the observed summary
statistic NPE needs, from the original run's simulation. NPE trains in
seconds and samples a posterior in milliseconds (Table 3b-equivalent for
this study), so there is no cost benefit to re-simulating -- only a
correctness benefit to *not* re-simulating, since every other column in the
file is already valid and re-deriving it a second time is a chance to
introduce a discrepancy for no reason.

Usage:
    python add_npe.py

Reads and overwrites results/logs/raw_replicates.csv (adding NPE columns)
and regenerates table1_recovery.csv and TABLES.md via the same aggregate()
function run_experiments.py uses.
"""

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd

from npe import train_npe, npe_summary
from run_experiments import aggregate
from paths import DATA, LOG_DIR, TABLE_DIR


def main():
    raw_path = LOG_DIR / "raw_replicates.csv"
    cfg_path = LOG_DIR / "experiment_config.json"
    df = pd.read_csv(raw_path)
    cfg = json.loads(cfg_path.read_text())
    # aggregate() reads cfg["truths"]/["J_grid"]/["with_sim"] as lists of
    # plain floats/ints; json round-trips TRUTHS as lists already, so no
    # conversion needed beyond what json.loads already gives back.

    print("training NPE...")
    posterior = train_npe(str(DATA))

    p1s, p2s, taus, cilen1, cilen2, cilent, cov1, cov2, covt, ess2, esst, secs = (
        [] for _ in range(12))
    for i, row in enumerate(df.itertuples(index=False)):
        seed = 10_000 * i + 1  # a fixed, simple per-row seed; see add_npe.py (1-D) for why
        t0 = time.time()
        s = npe_summary(posterior, row.obs, rng_seed=seed)
        secs.append(time.time() - t0)
        p1s.append(s["p1"]["mean"]); cilen1.append(s["p1"]["ci_len"])
        cov1.append(int(s["p1"]["ci_lo"] <= row.p1_true <= s["p1"]["ci_hi"]))
        p2s.append(s["p2"]["mean"]); cilen2.append(s["p2"]["ci_len"])
        cov2.append(int(s["p2"]["ci_lo"] <= row.p2_true <= s["p2"]["ci_hi"]))
        taus.append(s["tau"]["mean"]); cilent.append(s["tau"]["ci_len"])
        covt.append(int(s["tau"]["ci_lo"] <= row.tau_true <= s["tau"]["ci_hi"]))
        ess2.append(s["ess_p2"]); esst.append(s["ess_tau"])
        if (i + 1) % 10 == 0 or i + 1 == len(df):
            print(f"  [{i+1}/{len(df)}] scored")

    df["NPE_secs"] = secs
    df["NPE_acc"] = 1.0
    df["NPE_p1"] = p1s; df["NPE_p1_cilen"] = cilen1; df["NPE_p1_cov"] = cov1
    df["NPE_p2"] = p2s; df["NPE_p2_cilen"] = cilen2; df["NPE_p2_cov"] = cov2
    df["NPE_tau"] = taus; df["NPE_tau_cilen"] = cilent; df["NPE_tau_cov"] = covt
    df["NPE_ess_p2"] = ess2; df["NPE_ess_tau"] = esst
    df.to_csv(raw_path, index=False)
    print(f"wrote {raw_path}")

    tab = aggregate(df, cfg)
    tab.to_csv(TABLE_DIR / "table1_recovery.csv", index=False)
    print(f"wrote {TABLE_DIR}/table1_recovery.csv")

    lines = ["# 3-D two-stage model: parameter recovery\n",
             f"Config: `{json.dumps(cfg)}`\n",
             "`rmse_log` is RMSE in log10 units for p1/p2 (so 1.0 = off by an order of "
             "magnitude on average) and in absolute time units for tau. **Prefer it to "
             "`nrmse`**: where a parameter is weakly identified the posterior mean sits "
             "wherever the prior puts its mass, and natural-scale nRMSE then explodes "
             "without conveying anything.\n",
             "| truth (p1, p2, tau) | J | method | param | rmse_log | nRMSE | mean 95% CI width | coverage |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| ({r['p1']:.0e}, {r['p2']:.0e}, {r['tau']:.0f}) | {int(r['J'])} | "
                     f"{r['method']} | {r['param']} | {r['rmse_log']:.3f} | {r['nrmse']:.3f} | "
                     + (f"{r['ci_len']:.3e} | " if np.isfinite(r['ci_len']) else "- | ")
                     + (f"{r['coverage']:.2f} |" if np.isfinite(r['coverage']) else "- |"))
    (TABLE_DIR / "TABLES.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE_DIR}/TABLES.md")


if __name__ == "__main__":
    main()
