"""Add the NPE baseline to an already-completed accuracy sweep, without
rerunning exact ABC-MCMC / GPS-ABC / DNN-ABC.

`run_experiments.py`'s full sweep is dominated by the exact simulator inside
the ABC-MCMC baseline (hours, even parallelized); NPE is not (its training
takes seconds and its per-dataset posterior sampling takes milliseconds --
see Table 3b). Re-running the whole sweep just to add an NPE column would
mean paying that exact-simulator cost a second time for no reason: every
row in `results/logs/raw_replicates.csv` already stores `obs`, the observed
summary statistic NPE needs, from the original run's simulation. This script
trains NPE (as `run_experiments.py` would, on the same replicate-1-8 rows)
and scores it against those already-simulated observations directly,
skipping every backend that's already there.

Usage:
    python add_npe.py

Reads and overwrites results/logs/raw_replicates.csv (adding NPE columns to
the existing rows) and regenerates every table that depends on it --
table1_mse.csv, table2_cilength.csv, table_coverage.csv, TABLES.md, and
mcse.md -- via the same aggregation functions `run_experiments.py` uses, so
there is exactly one implementation of each to keep in sync, not two.
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd

from npe import train_npe_by_J, npe_point_and_interval
from run_experiments import aggregate_tables, aggregate_coverage, fmt_table1, fmt_table2, fmt_coverage
from paths import DATA, LOG_DIR, TABLE_DIR


def main():
    raw_path = LOG_DIR / "raw_replicates.csv"
    cfg_path = LOG_DIR / "experiment_config.json"
    df = pd.read_csv(raw_path)
    cfg = json.loads(cfg_path.read_text())
    J_grid = sorted(df["J"].unique().tolist())

    print(f"training NPE on {J_grid}...")
    posterior_by_J = train_npe_by_J(str(DATA), J_grid)

    p_hats, ci_los, ci_his, ci_lens = [], [], [], []
    for i, row in enumerate(df.itertuples(index=False)):
        # A task-specific seed in the same style _one_replicate uses for its
        # own per-task rng (10_000 per -log10(p), 100 per J, 1 per rep), offset
        # so it doesn't collide with any seed a from-scratch run would derive
        # for this task's other backends. Reproducibility of *this script*,
        # not bit-identical replication of a hypothetical unified run's rng
        # stream -- that would require re-running the simulator this script
        # exists to avoid, since fluc_exp's own internal draws aren't
        # reproduced here.
        npe_seed = 10_000 * int(round(-np.log10(row.p_true))) + 100 * int(row.J) + int(row.rep) + 1
        p_hat, ci_lo, ci_hi, ci_len = npe_point_and_interval(
            posterior_by_J[int(row.J)], row.obs, rng_seed=npe_seed)
        p_hats.append(p_hat); ci_los.append(ci_lo); ci_his.append(ci_hi); ci_lens.append(ci_len)
        if (i + 1) % 50 == 0 or i + 1 == len(df):
            print(f"  [{i+1}/{len(df)}] scored")

    df["NPE"] = p_hats
    df["NPE_ci_lo"] = ci_los
    df["NPE_ci_hi"] = ci_his
    df["NPE_cilen"] = ci_lens
    df.to_csv(raw_path, index=False)
    print(f"wrote {raw_path}")

    t1, t2 = aggregate_tables(df, cfg)
    t1.to_csv(TABLE_DIR / "table1_mse.csv", index=False)
    t2.to_csv(TABLE_DIR / "table2_cilength.csv", index=False)
    tcov = aggregate_coverage(df, cfg)
    tcov.to_csv(TABLE_DIR / "table_coverage.csv", index=False)
    print(f"wrote {TABLE_DIR}/table1_mse.csv, table2_cilength.csv, table_coverage.csv")

    tbl1_md, tbl2_md, tblcov_md = fmt_table1(t1), fmt_table2(t2), fmt_coverage(tcov)
    tables_md = TABLE_DIR / "TABLES.md"
    text = tables_md.read_text()
    # Replace just the Table 1/2/2b sections; Table 3/3b (timing) are untouched.
    import re
    text = re.sub(r"(## Table 1 - .*?\n\n).*?(\n\n## Table 2 - )", r"\1" + tbl1_md.replace("\\", "\\\\") + r"\2", text, flags=re.S)
    text = re.sub(r"(## Table 2 - .*?\n\n).*?(\n\n## Table 2b - )", r"\1" + tbl2_md.replace("\\", "\\\\") + r"\2", text, flags=re.S)
    text = re.sub(r"(## Table 2b - .*?\n\n).*?(\n\n## Table 3 - )", r"\1" + tblcov_md.replace("\\", "\\\\") + r"\2", text, flags=re.S)
    tables_md.write_text(text)
    print(f"wrote {tables_md}")


if __name__ == "__main__":
    main()
