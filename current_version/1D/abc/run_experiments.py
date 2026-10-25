"""Reproduce Table 1 (MSE), Table 2 (95% interval length) and Table 3 (compute
time) from Lu, Zhu & Wu (2023), adding a DNN-ABC column.

For each (p, J) in {1e-4,1e-3,1e-2} x {10,50,100} and each replicate: simulate
observed data (exact simulator), estimate p with MOM, MLE, ABC-MCMC (exact
sim in the loop), GPS-ABC (GP surrogate), DNN-ABC (our MLP surrogate), then
aggregate errors/interval-lengths per cell. All ABC-MCMC methods share the
same sampler/prior/proposal/eps -- only how the summary statistic is obtained
differs. See mcse.py for whether each table entry is signal or replicate
noise. `Pool`: replicates run in parallel, each seeded from its own
(p, J, rep), so results don't depend on core count.

Usage: python run_experiments.py --reps 8 --nmcmc 600 --burnin 250 --ns 8
"""

import argparse
import json
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

# --- make sibling code folders + paths.py importable (package uses flat imports) ---
_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc", _ROOT / "figures"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd

from simulator import solve_tp, fluc_exp, summary_stat
from estimators import estimate_mom, estimate_mle
from abc_mcmc import run_abc_mcmc, point_and_interval
from surrogates import fit_gp_surrogate
from train import load_surrogate, load_splits, run as train_run
from paths import DATA, RESULTS, TABLE_DIR, FIG_DIR, MODEL_DIR, LOG_DIR
# Lower bound -5: extending toward p=1e-8 lets the exact-simulator baseline
# wander into exponential-cost territory (one call can take minutes). Upper
# bound -1.5, not -2.0: -2.0 exactly coincided with log10(1e-2), the largest
# tested p, so the prior forbade proposing theta above the truth there and
# every method measured ~0% coverage at that cell (grid artifact, not an
# estimator defect -- see aggregate_coverage's docstring). -1.5 gives every
# tested p real headroom from the boundary.
PRIOR_RANGE = (-5.0, -1.5)

_G = {}  # per-worker cache for the fitted surrogates, populated by _init_worker


def _ckpt_path_for_J(base_ckpt_path, J):
    """Path convention shared with network/train.py's run(): the deployed
    J=100 checkpoint is unsuffixed, every other J gets its own file."""
    base_ckpt_path = Path(base_ckpt_path)
    if J == 100:
        return base_ckpt_path
    return base_ckpt_path.with_name(f"{base_ckpt_path.stem}_J{J}{base_ckpt_path.suffix}")


def _fit_gps_by_J(data_path, J_grid):
    """Fit the GPS-ABC baseline once per J, in the main process, shared
    read-only across every worker. Refitting independently per worker used to
    make startup alone take hours (30 workers x O(n^3) GP fit each); fitting
    once here and handing it to `_init_worker` removes that multiplier.
    """
    gp_by_J = {}
    for J in J_grid:
        (x_tr, y_tr), (x_va, y_va), _ = load_splits(data_path, J=(None if J == 100 else J))
        x = np.concatenate([x_tr, x_va])
        y = np.concatenate([y_tr, y_va])
        gp_by_J[J] = fit_gp_surrogate(x, y, budget=None)
    return gp_by_J


def _init_worker(ckpt_path, gp_by_J, cfg):
    """Runs once per worker: load the DNN for every J (cheap) and stash the
    already-fitted GPs from `_fit_gps_by_J`, so `_one_replicate` can score
    each replicate against a surrogate that knows its culture count.

    Also pins BLAS (OpenMP/OpenBLAS/MKL) to 1 thread per worker -- otherwise
    every worker spawns its own full-width BLAS thread pool for small
    numpy/GP calls, oversubscribing the machine (looks "Running" with real
    CPU time but makes near-zero progress; process- and thread-level
    parallelism fighting for the same cores). The Pool already supplies all
    the parallelism needed, so threads lose.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import threadpoolctl
    threadpoolctl.threadpool_limits(1)
    dnn_by_J = {J: load_surrogate(str(_ckpt_path_for_J(ckpt_path, J))) for J in cfg["J_grid"]}
    _G["dnn_by_J"], _G["gp_by_J"], _G["cfg"] = dnn_by_J, gp_by_J, cfg


def _clamp_init(p_hat):
    """Turn a point estimate into a valid MCMC starting value on the
    theta = log10(p) scale, kept just inside the prior bounds (falls back to
    the prior midpoint if p_hat is missing or non-positive)."""
    lo, hi = PRIOR_RANGE
    th = np.log10(max(p_hat, 1e-8)) if (p_hat is not None and np.isfinite(p_hat) and p_hat > 0) else 0.5 * (lo + hi)
    return float(min(max(th, lo + 1e-6), hi - 1e-6))


def _one_replicate(task):
    """Simulate one data set at `task = (p_true, J, rep)`, run all five
    estimators on it, and return one results row.
    """
    p_true, J, rep = task
    cfg = _G["cfg"]
    dnn, gp = _G["dnn_by_J"][J], _G["gp_by_J"][J]
    rng = np.random.default_rng(10_000 * int(round(-np.log10(p_true))) + 100 * J + rep)

    tp = solve_tp(1, 1, p_true, 20)
    Z_vec, X_vec = fluc_exp(1, 1, 1, p_true, tp, J, rng, use_slow=True)
    obs = summary_stat(Z_vec, X_vec)

    p_mom = estimate_mom(Z_vec, X_vec)
    p_mle = estimate_mle(Z_vec, X_vec)
    th0 = _clamp_init(p_mom)

    tp_fn = lambda th: solve_tp(1, 1, 10.0 ** th, 20)
    sim_kwargs = dict(Z0=1, a=1, delta=1, J=J, tp_fn=tp_fn, use_slow=True, method="synthetic")

    out = {"p_true": p_true, "J": J, "rep": rep, "obs": obs,
           "MOM": p_mom, "MLE": p_mle}
    backends = [
        ("ABC-MCMC", dict(backend="sim", sim_kwargs=sim_kwargs, ns=cfg["ns"])),
        ("GPS-ABC", dict(backend="gp", surrogate=gp)),
        ("DNN-ABC", dict(backend="dnn", surrogate=dnn)),
    ]
    for name, kw in backends:
        samples, acc = run_abc_mcmc(
            obs, n_mcmc=cfg["nmcmc"], theta_init=th0, s=0.15,
            rng=np.random.default_rng(rng.integers(2**63 - 1)),
            prior_range=PRIOR_RANGE, lam=2.0, eps=cfg["eps"], **kw)
        p_hat, ci_lo, ci_hi, ci_len = point_and_interval(samples, cfg["burnin"])
        out[name] = p_hat
        out[name + "_cilen"] = ci_len
        out[name + "_ci_lo"] = ci_lo
        out[name + "_ci_hi"] = ci_hi
        out[name + "_acc"] = acc
    return out


def run_accuracy(cfg, ckpt_path):
    """Run `_one_replicate` over the whole (p, J, replicate) grid in
    parallel; one DataFrame row per replicate, which Tables 1 and 2 are
    then aggregated from.
    """
    tasks = [(p, J, r) for p in cfg["p_grid"] for J in cfg["J_grid"]
             for r in range(cfg["reps"])]
    n = len(tasks)
    print(f"fitting GPS-ABC baseline (once per J, shared across workers)...", flush=True)
    t_gp = time.time()
    gp_by_J = _fit_gps_by_J(str(DATA), cfg["J_grid"])
    print(f"  done in {(time.time()-t_gp)/60:.1f}m", flush=True)
    print(f"accuracy: {n} tasks ({len(cfg['p_grid'])}p x {len(cfg['J_grid'])}J x {cfg['reps']} reps) "
          f"on {cfg['workers']} workers", flush=True)
    rows = []
    t0 = time.time()
    with Pool(cfg["workers"], initializer=_init_worker,
              initargs=(ckpt_path, gp_by_J, cfg)) as pool:
        # imap_unordered so we can log real progress + ETA (pool.map is opaque)
        for i, r in enumerate(pool.imap_unordered(_one_replicate, tasks), 1):
            rows.append(r)
            if i % 10 == 0 or i == n:
                el = time.time() - t0
                eta = el / i * (n - i)
                print(f"  [{i}/{n}] done  elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
    return pd.DataFrame(rows)


def aggregate_tables(df, cfg):
    """Collapse per-replicate results into Table 1 (MSE and normalized RMSE
    per (p, J) cell) and Table 2 (mean 95% credible-interval length)."""
    methods = ["MOM", "MLE", "ABC-MCMC", "GPS-ABC", "DNN-ABC"]
    t1, t2 = [], []
    for p in cfg["p_grid"]:
        for J in cfg["J_grid"]:
            sub = df[(df["p_true"] == p) & (df["J"] == J)]
            row1 = {"p": p, "J": J}
            row2 = {"p": p, "J": J}
            for m in methods:
                est = sub[m].to_numpy(dtype=float)
                est = est[np.isfinite(est)]
                mse = float(np.mean((est - p) ** 2)) if len(est) else np.nan
                row1[m] = mse
                row1[m + "_nrmse"] = np.sqrt(mse) / p if np.isfinite(mse) else np.nan
                if m + "_cilen" in sub:
                    cl = sub[m + "_cilen"].to_numpy(dtype=float)
                    cl = cl[np.isfinite(cl)]
                    row2[m] = float(np.mean(cl)) if len(cl) else np.nan
            t1.append(row1)
            t2.append(row2)
    return pd.DataFrame(t1), pd.DataFrame(t2)


def aggregate_coverage(df, cfg):
    """Empirical POSTERIOR coverage of the 95% ABC credible interval, per (p,
    J) cell, for the three ABC backends. Distinct from the surrogate's own
    REGRESSION coverage on held-out data (does mean +/- z*sd bracket the true
    log10(d_bar)?) -- the two are different random variables and can diverge;
    this measures the one that matters for Table 2. Binomial MCSE attached
    per cell so a departure from 0.95 can be judged against sampling noise.
    """
    methods = ["ABC-MCMC", "GPS-ABC", "DNN-ABC"]
    rows = []
    for p in cfg["p_grid"]:
        for J in cfg["J_grid"]:
            sub = df[(df["p_true"] == p) & (df["J"] == J)]
            row = {"p": p, "J": J, "R": len(sub)}
            for m in methods:
                lo_col, hi_col = m + "_ci_lo", m + "_ci_hi"
                if lo_col not in sub or hi_col not in sub:
                    continue  # older raw_replicates.csv without endpoints saved
                lo = sub[lo_col].to_numpy(dtype=float)
                hi = sub[hi_col].to_numpy(dtype=float)
                ok = np.isfinite(lo) & np.isfinite(hi)
                lo, hi, n = lo[ok], hi[ok], int(ok.sum())
                cov = float(np.mean((lo <= p) & (p <= hi))) if n else np.nan
                row[m + "_coverage"] = cov
                row[m + "_coverage_mcse"] = (
                    float(np.sqrt(cov * (1 - cov) / n)) if n and np.isfinite(cov) else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def run_timing(cfg, ckpt_path):
    """Single-process seconds / 100 MCMC iterations for each ABC method."""
    import warnings
    warnings.filterwarnings("ignore")
    dnn_by_J, gp_by_J = {}, {}
    for J in cfg["J_grid"]:
        dnn_by_J[J] = load_surrogate(str(_ckpt_path_for_J(ckpt_path, J)))
        (x_tr, y_tr), (x_va, y_va), _ = load_splits(str(DATA), J=(None if J == 100 else J))
        gp_by_J[J] = fit_gp_surrogate(np.concatenate([x_tr, x_va]), np.concatenate([y_tr, y_va]),
                                      budget=None)
    n_time = cfg["timing_iters"]
    rows = []
    for p in cfg["p_grid"]:
        for J in cfg["J_grid"]:
            dnn, gp = dnn_by_J[J], gp_by_J[J]
            rng = np.random.default_rng(0)
            tp = solve_tp(1, 1, p, 20)
            obs = summary_stat(*fluc_exp(1, 1, 1, p, tp, J, rng, use_slow=True))
            th0 = -np.log10(1.0) - (-np.log10(p))  # = log10(p)
            tp_fn = lambda th: solve_tp(1, 1, 10.0 ** th, 20)
            sk = dict(Z0=1, a=1, delta=1, J=J, tp_fn=tp_fn, use_slow=True, method="synthetic")
            per = {"p": p, "J": J}
            for name, kw in [("ABC-MCMC", dict(backend="sim", sim_kwargs=sk, ns=cfg["ns"])),
                             ("GPS-ABC", dict(backend="gp", surrogate=gp)),
                             ("DNN-ABC", dict(backend="dnn", surrogate=dnn))]:
                t = time.time()
                run_abc_mcmc(obs, n_mcmc=n_time, theta_init=th0, eps=cfg["eps"],
                             rng=np.random.default_rng(1), prior_range=PRIOR_RANGE, **kw)
                per[name] = (time.time() - t) / n_time * 100.0  # sec / 100 iters
            rows.append(per)
            print(f"  timed p={p:.0e} J={J}: ABC-MCMC={per['ABC-MCMC']:.2f}s "
                  f"GPS-ABC={per['GPS-ABC']:.3f}s DNN-ABC={per['DNN-ABC']:.3f}s per 100 iter")
    return pd.DataFrame(rows)


def fmt_table1(t1, methods=("MOM", "MLE", "ABC-MCMC", "GPS-ABC", "DNN-ABC")):
    lines = ["| p | J | " + " | ".join(methods) + " |",
             "|---|---|" + "|".join(["---"] * len(methods)) + "|"]
    for _, r in t1.iterrows():
        cells = []
        for m in methods:
            cells.append(f"{r[m]:.2e} ({r[m+'_nrmse']:.2f})")
        lines.append(f"| {r['p']:.0e} | {int(r['J'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_coverage(tcov, methods=("ABC-MCMC", "GPS-ABC", "DNN-ABC")):
    """Posterior coverage table: does the 95% ABC credible interval actually
    bracket the true p? (A different quantity than regression coverage -- see
    aggregate_coverage's docstring.)"""
    lines = ["| p | J | R | " + " | ".join(f"{m} cov (MCSE)" for m in methods) + " |",
             "|---|---|---|" + "|".join(["---"] * len(methods)) + "|"]
    for _, r in tcov.iterrows():
        cells = []
        for m in methods:
            c, s = r.get(m + "_coverage"), r.get(m + "_coverage_mcse")
            cells.append(f"{c:.3f} ({s:.3f})" if pd.notna(c) else "-")
        lines.append(f"| {r['p']:.0e} | {int(r['J'])} | {int(r['R'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_table2(t2, methods=("MOM", "MLE", "ABC-MCMC", "GPS-ABC", "DNN-ABC")):
    lines = ["| p | J | " + " | ".join(methods) + " |",
             "|---|---|" + "|".join(["---"] * len(methods)) + "|"]
    for _, r in t2.iterrows():
        cells = [f"{r[m]:.2e}" if m in r and np.isfinite(r[m]) else "-" for m in methods]
        lines.append(f"| {r['p']:.0e} | {int(r['J'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_table3(t3):
    methods = ["ABC-MCMC", "GPS-ABC", "DNN-ABC"]
    lines = ["| p | J | " + " | ".join(m + " (s/100it)" for m in methods) + " | ABC/DNN speedup |",
             "|---|---|---|---|---|---|"]
    for _, r in t3.iterrows():
        speed = r["ABC-MCMC"] / r["DNN-ABC"] if r["DNN-ABC"] > 0 else np.nan
        lines.append(f"| {r['p']:.0e} | {int(r['J'])} | {r['ABC-MCMC']:.2f} | "
                     f"{r['GPS-ABC']:.3f} | {r['DNN-ABC']:.3f} | {speed:.0f}x |")
    return "\n".join(lines)


def timing_plot(t3, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [f"{r['p']:.0e}\nJ={int(r['J'])}" for _, r in t3.iterrows()]
    xpos = np.arange(len(labels))
    w = 0.27
    ax.bar(xpos - w, t3["ABC-MCMC"], w, label="ABC-MCMC", color="tab:red")
    ax.bar(xpos, t3["GPS-ABC"], w, label="GPS-ABC (GP)", color="tab:green")
    ax.bar(xpos + w, t3["DNN-ABC"], w, label="DNN-ABC (ours)", color="tab:blue")
    ax.set_yscale("log")
    ax.set_ylabel("seconds / 100 MCMC iterations (log scale)")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Per-iteration cost: ABC-MCMC vs. surrogate ABC")
    ax.legend(); fig.tight_layout()
    fig.savefig(outpath, dpi=150); plt.close(fig)


def main():
    """CLI entry point: (re)train the surrogate if needed -> Phase A accuracy
    sweep -> Phase B timing sweep -> write the CSVs and TABLES.md.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--nmcmc", type=int, default=600)
    ap.add_argument("--burnin", type=int, default=250)
    ap.add_argument("--ns", type=int, default=8, help="sims per ABC-MCMC iteration")
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--timing-iters", type=int, default=60)
    ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 2) - 2))
    ap.add_argument("--p-grid", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--J-grid", type=int, nargs="+", default=[10, 50, 100])
    ap.add_argument("--retrain", action="store_true", help="retrain the DNN surrogate first")
    args = ap.parse_args()

    ckpt = MODEL_DIR / "surrogate_1d.pt"
    # One DNN surrogate per J in the grid, not just the deployed J=100 one --
    # see _init_worker's docstring and Section on posterior coverage / README
    # Section 4.3b for why: a single J=100-trained surrogate queried at every
    # J is what produced the measured undercoverage at small J.
    for J in args.J_grid:
        j_ckpt = _ckpt_path_for_J(ckpt, J)
        if args.retrain or not j_ckpt.exists():
            print(f"training DNN surrogate at J={J}...")
            train_run(str(DATA), J=(None if J == 100 else J))

    cfg = dict(reps=args.reps, nmcmc=args.nmcmc, burnin=args.burnin, ns=args.ns,
               eps=args.eps, timing_iters=args.timing_iters, workers=args.workers,
               p_grid=list(args.p_grid), J_grid=list(args.J_grid))
    with open(LOG_DIR / "experiment_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    t_start = time.time()
    print("\n=== Phase A: accuracy (Table 1 & 2) ===")
    raw = run_accuracy(cfg, str(ckpt))
    raw.to_csv(LOG_DIR / "raw_replicates.csv", index=False)
    t1, t2 = aggregate_tables(raw, cfg)
    t1.to_csv(TABLE_DIR / "table1_mse.csv", index=False)
    t2.to_csv(TABLE_DIR / "table2_cilength.csv", index=False)
    tcov = aggregate_coverage(raw, cfg)
    tcov.to_csv(TABLE_DIR / "table_coverage.csv", index=False)

    print("\n=== Phase B: timing (Table 3) ===")
    t3 = run_timing(cfg, str(ckpt))
    t3.to_csv(TABLE_DIR / "table3_timing.csv", index=False)
    timing_plot(t3, FIG_DIR / "table3_timing.png")

    tbl1_md = fmt_table1(t1)
    tbl2_md = fmt_table2(t2)
    tbl3_md = fmt_table3(t3)
    tblcov_md = fmt_coverage(tcov)
    with open(TABLE_DIR / "TABLES.md", "w") as f:
        f.write("# Reproduced tables with DNN-ABC column\n\n")
        f.write(f"Config: {json.dumps(cfg)}\n\n")
        f.write("## Table 1 - MSE of p-hat, and (sqrt(MSE)/p) in parentheses\n\n")
        f.write(tbl1_md + "\n\n")
        f.write("## Table 2 - mean 95% interval length of p-hat\n\n")
        f.write(tbl2_md + "\n\n")
        f.write("## Table 2b - POSTERIOR coverage of the 95% credible interval\n\n")
        f.write("Does the interval actually bracket the true p 95% of the time? "
                "(Distinct from Section sec:sim1D_calib's regression coverage --\n"
                "see aggregate_coverage's docstring in run_experiments.py.)\n\n")
        f.write(tblcov_md + "\n\n")
        f.write("## Table 3 - seconds per 100 MCMC iterations\n\n")
        f.write(tbl3_md + "\n")
    print("\n" + tbl1_md + "\n\n" + tblcov_md + "\n\n" + tbl3_md)
    print(f"\ntotal wall time: {(time.time()-t_start)/60:.1f} min")
    print(f"results written to {RESULTS}")


if __name__ == "__main__":
    main()
