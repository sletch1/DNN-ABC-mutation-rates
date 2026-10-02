"""Regenerate the 3-D two-stage-mutation training design at a larger budget.

Reproduces the schema of data/slow_data_3D.csv (Z0,a,delta,p1,p2,tau,tp,J,
design,rep,d_bar,d_1..d_100) but over a 10,000-point Latin-hypercube design
instead of the original 2,000, with the same 10 replicates per point, J=100,
tp=10, a=1, delta=1, and the exact ("slow") simulator.

Bounds (from the manuscript, Section on Study II ground truth):
    log10 p1, log10 p2 in [-5, -1.3]
    tau               in [0.1, 9.9]

Parallelized across (design, replicate) tasks with a process pool (same
pattern as abc/run_experiments.py, abc/add_exact.py), since each task is an
independent simulator call with no shared state.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "abc"))
from simulator import fluc_exp_2stage, summary_stat  # noqa: E402

N_DESIGN = 10_000
N_REP = 10
J = 100
TP = 10.0
A = 1.0
DELTA = 1.0
Z0 = 1
LOG_P_LO, LOG_P_HI = -5.0, -1.3
TAU_LO, TAU_HI = 0.1, 9.9
SEED = 20240925  # fixed, for a reproducible design
N_WORKERS = 30
OUT_PATH = Path(__file__).resolve().parent / "slow_data_3D_n10000.csv"


def build_design():
    sampler = qmc.LatinHypercube(d=3, seed=SEED)
    u = sampler.random(N_DESIGN)  # (N_DESIGN, 3) in [0,1)^3
    log_p1 = LOG_P_LO + u[:, 0] * (LOG_P_HI - LOG_P_LO)
    log_p2 = LOG_P_LO + u[:, 1] * (LOG_P_HI - LOG_P_LO)
    tau = TAU_LO + u[:, 2] * (TAU_HI - TAU_LO)
    p1 = 10 ** log_p1
    p2 = 10 ** log_p2
    return p1, p2, tau


def _run_one(args):
    design_idx, rep_idx, p1, p2, tau = args
    # deterministic, unique seed per (design, replicate) task
    seed = SEED * 1_000_003 + design_idx * 100 + rep_idx
    rng = np.random.default_rng(seed)
    Z_vec, X_vec = fluc_exp_2stage(
        Z0, A, p1, p2, tau, TP, J, rng, use_slow=True, delta=DELTA
    )
    d_bar = summary_stat(Z_vec, X_vec)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_i = (X_vec / Z_vec) ** (1.0 / 4.0)
    d_i = np.where(np.isfinite(d_i), d_i, 0.0)

    row = {
        "Z0": Z0, "a": A, "delta": DELTA,
        "p1": p1, "p2": p2, "tau": tau, "tp": TP, "J": J,
        "design": design_idx, "rep": rep_idx, "d_bar": d_bar,
    }
    for k in range(J):
        row[f"d_{k+1}"] = d_i[k]
    return row


def main():
    p1_arr, p2_arr, tau_arr = build_design()

    tasks = []
    for design_idx in range(1, N_DESIGN + 1):
        p1, p2, tau = p1_arr[design_idx - 1], p2_arr[design_idx - 1], tau_arr[design_idx - 1]
        for rep_idx in range(1, N_REP + 1):
            tasks.append((design_idx, rep_idx, p1, p2, tau))

    print(f"Total tasks: {len(tasks)} ({N_DESIGN} design points x {N_REP} reps), "
          f"{N_WORKERS} workers", flush=True)

    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS) as pool:
        for i, row in enumerate(pool.imap_unordered(_run_one, tasks, chunksize=20)):
            rows.append(row)
            if (i + 1) % 5000 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - (i + 1)) / rate
                print(f"  {i+1}/{len(tasks)} done, {elapsed:.0f}s elapsed, "
                      f"ETA {eta:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"Generation done in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df = df.sort_values(["design", "rep"]).reset_index(drop=True)
    cols = ["Z0", "a", "delta", "p1", "p2", "tau", "tp", "J", "design", "rep", "d_bar"] + \
           [f"d_{k+1}" for k in range(J)]
    df = df[cols]
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
