"""Standalone GP-vs-DNN cost-scaling benchmark.

Purpose: directly demonstrate the O(n^3) GP-fitting / O(n) GP-query cost
claim against the DNN's flat query cost, at design budgets well beyond the
n=300 used in the main Study II recovery tables. This is a TIMING-ONLY
measurement -- it does not touch, and is not meant to inform, the accuracy
tables. Reuses the already-generated, already-corrected (root-2 convention)
n=10,000-point dataset at data/slow_data_3D.csv, so no new simulator calls
are needed.

Usage:
    python gp_cost_scaling.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "abc"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "network"))

from surrogates import fit_gp_surrogate_3d, DNNSurrogate3D  # noqa: E402
from train import load_surrogate, TEST_REPS, summary_from_cultures  # noqa: E402
from paths import DATA, MODEL_DIR  # noqa: E402

BUDGETS = [300, 1000, 3000, 10000]
N_QUERY = 100          # matches Table 3's "per 100 iterations" convention
N_QUERY_REPEATS = 5    # repeat the query timing to get a stable estimate
SEED = 0


def main():
    torch.set_num_threads(1)

    df = pd.read_csv(DATA)
    tr = df[~df["rep"].isin(TEST_REPS)].reset_index(drop=True)
    X = np.column_stack([np.log10(tr.p1), np.log10(tr.p2), tr.tau]).astype(np.float64)
    S = summary_from_cultures(tr)
    y = np.log10(S)

    rng = np.random.default_rng(SEED)
    query_idx = rng.choice(len(X), size=N_QUERY, replace=False)
    X_query = X[query_idx]

    print(f"Pool size available for GP budget sampling: {len(X)} rows")
    print(f"{'budget':>8} {'fit_s':>10} {'query_s_per_100':>18}")

    results = []
    for budget in BUDGETS:
        t0 = time.perf_counter()
        gp = fit_gp_surrogate_3d(X, y, budget=budget, seed=SEED)
        fit_s = time.perf_counter() - t0

        q_times = []
        for _ in range(N_QUERY_REPEATS):
            t0 = time.perf_counter()
            gp.predict(X_query)
            q_times.append(time.perf_counter() - t0)
        query_s = float(np.median(q_times))

        print(f"{budget:>8} {fit_s:>10.3f} {query_s:>18.4f}")
        results.append({"method": "GPS-ABC", "budget": budget,
                         "fit_seconds": fit_s, "query_seconds_per_100": query_s})

    # DNN reference: flat, budget-independent query cost
    ckpt = MODEL_DIR / "surrogate_3d.pt"
    dnn = load_surrogate(str(ckpt))
    q_times = []
    for _ in range(N_QUERY_REPEATS):
        t0 = time.perf_counter()
        dnn.predict(X_query)
        q_times.append(time.perf_counter() - t0)
    dnn_query_s = float(np.median(q_times))
    print(f"{'DNN-ABC':>8} {'--':>10} {dnn_query_s:>18.4f}   (flat, independent of budget)")
    results.append({"method": "DNN-ABC", "budget": None,
                     "fit_seconds": None, "query_seconds_per_100": dnn_query_s})

    out = pd.DataFrame(results)
    out_path = Path(__file__).resolve().parent / "gp_cost_scaling_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
