"""Reproduce Table 1 (MSE), Table 2 (95% interval length) and Table 3 (compute
time) in the 3-D (p, a, delta) regime, adding a DNN-ABC column alongside the
paper's MOM/MLE, ABC-MCMC and GPS-ABC columns.

Experimental design (constant-rate model, inference on p at KNOWN (a, delta)):
  truth  = exact/slow simulator (Algorithm 2), matching the DNN's training data.
  grid   = p in {1e-3, 1e-2} x (a, delta) regimes x J in {50, 100}.
  For each (p, a, delta, J) and each replicate:
     - simulate observed fluctuation data at (p, a, delta) (slow sim),
       obs = mean sqrt(X/Z),
     - estimate p with: MOM, MLE (both a-, delta-agnostic -> a fair "naive"
       baseline), ABC-MCMC (slow sim inside the loop, given a, delta),
       GPS-ABC (GP surrogate capped at 300 points), DNN-ABC (our residual MLP,
       all ~5000 rows).
  Aggregate over replicates: Table 1 -> MSE and sqrt(MSE)/p ; Table 2 -> mean 95%
  interval length. Table 3 (timing) times each method's seconds/100 MCMC iters.

The surrogates are queried at the true (a, delta) each iteration; only the way the
summary statistic and its uncertainty are obtained changes across the ABC columns.

Usage:
    python run_experiments.py --reps 24 --nmcmc 600 --burnin 250 --ns 6
"""

import argparse
import json
import sys
import time
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
from surrogates import fit_gp_surrogate_3d
from train import load_surrogate, run as train_run
from paths import DATA, RESULTS, TABLE_DIR, FIG_DIR, MODEL_DIR, LOG_DIR

# Prior bounded so the ABC-MCMC baseline's slow simulator stays feasible while
# bracketing both truths (p in {1e-3, 1e-2}) in the interior. Surrogates are
# trained on the full log10(p) in [-8,-2] and only queried in-range.
PRIOR_RANGE = (-4.0, -1.5)
GP_BUDGET = 300  # the GPS-ABC ceiling in 3-D (space-filling); DNN uses all ~5000 rows

# (a, delta) regimes: paper's constant-rate point, both-elevated, slow-mutant.
REGIMES = [(1.0, 1.0), (1.5, 1.5), (1.0, 0.5)]

_G = {}


def _init_worker(ckpt_path, data_path, cfg):
    import warnings
    warnings.filterwarnings("ignore")
    dnn = load_surrogate(ckpt_path)
    df = pd.read_csv(data_path)
    tr = df[df["rep"].isin([1, 2, 3, 4, 5, 6, 7, 8])]
    X = np.column_stack([np.log10(tr["p"].to_numpy()), tr["a"].to_numpy(),
                         tr["delta"].to_numpy()])
    y = np.log10(tr["d_bar"].to_numpy())
    gp = fit_gp_surrogate_3d(X, y, budget=cfg["gp_budget"])
    _G["dnn"], _G["gp"], _G["cfg"] = dnn, gp, cfg


def _clamp_init(p_hat):
    lo, hi = PRIOR_RANGE
    th = np.log10(max(p_hat, 1e-8)) if (p_hat is not None and np.isfinite(p_hat) and p_hat > 0) else 0.5 * (lo + hi)
    return float(min(max(th, lo + 1e-6), hi - 1e-6))


def _one_replicate(task):
    p_true, a, delta, J, rep = task
    cfg = _G["cfg"]
    dnn, gp = _G["dnn"], _G["gp"]
    seed = (10_000 * int(round(-np.log10(p_true))) + 1000 * int(round(a * 10))
            + 100 * int(round(delta * 10)) + J + rep)
    rng = np.random.default_rng(seed)

    tp = solve_tp(1, a, p_true, 20)
    Z_vec, X_vec = fluc_exp(1, a, delta, p_true, tp, J, rng, use_slow=True)
    obs = summary_stat(Z_vec, X_vec)

    p_mom = estimate_mom(Z_vec, X_vec)
    p_mle = estimate_mle(Z_vec, X_vec)
    th0 = _clamp_init(p_mom)

    tp_fn = lambda th: solve_tp(1, a, 10.0 ** th, 20)
    sim_kwargs = dict(Z0=1, J=J, tp_fn=tp_fn, use_slow=True)

    out = {"p_true": p_true, "a": a, "delta": delta, "J": J, "rep": rep,
           "obs": obs, "MOM": p_mom, "MLE": p_mle}
    backends = [
        ("ABC-MCMC", dict(backend="sim", sim_kwargs=sim_kwargs, ns=cfg["ns"])),
        ("GPS-ABC", dict(backend="gp", surrogate=gp)),
        ("DNN-ABC", dict(backend="dnn", surrogate=dnn)),
    ]
    for name, kw in backends:
        samples, acc = run_abc_mcmc(
            obs, a_known=a, delta_known=delta, n_mcmc=cfg["nmcmc"], theta_init=th0,
            s=0.15, rng=np.random.default_rng(rng.integers(2**63 - 1)),
            prior_range=PRIOR_RANGE, lam=2.0, eps=cfg["eps"], **kw)
        p_hat, ci_lo, ci_hi, ci_len = point_and_interval(samples, cfg["burnin"])
        out[name] = p_hat
        out[name + "_cilen"] = ci_len
        out[name + "_cilo"] = ci_lo
        out[name + "_cihi"] = ci_hi
        out[name + "_acc"] = acc
    return out


def run_accuracy(cfg, ckpt_path):
    tasks = [(p, a, d, J, r) for p in cfg["p_grid"] for (a, d) in cfg["regimes"]
             for J in cfg["J_grid"] for r in range(cfg["reps"])]
    # Expensive-tasks-first (longest-processing-time) so dynamic scheduling never
    # leaves cores idle waiting on one big job at the tail. Per-task cost of the
    # ABC-MCMC baseline ~ J * exp(a*tp) (simulator population x cultures).
    tasks.sort(key=lambda t: t[3] * np.exp(t[1] * solve_tp(1, t[1], t[0], 20)),
               reverse=True)
    n = len(tasks)
    print(f"accuracy: {n} tasks ({len(cfg['p_grid'])}p x {len(cfg['regimes'])} regimes "
          f"x {len(cfg['J_grid'])}J x {cfg['reps']} reps) on {cfg['workers']} workers",
          flush=True)
    rows = []
    t0 = time.time()
    with Pool(cfg["workers"], initializer=_init_worker,
              initargs=(ckpt_path, str(DATA), cfg)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one_replicate, tasks), 1):
            rows.append(r)
            if i % 10 == 0 or i == n:
                el = time.time() - t0
                eta = el / i * (n - i)
                print(f"  [{i}/{n}] done  elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
    return pd.DataFrame(rows)


def aggregate_tables(df, cfg):
    methods = ["MOM", "MLE", "ABC-MCMC", "GPS-ABC", "DNN-ABC"]
    t1, t2 = [], []
    for p in cfg["p_grid"]:
        for (a, d) in cfg["regimes"]:
            for J in cfg["J_grid"]:
                sub = df[(df["p_true"] == p) & (df["a"] == a) & (df["delta"] == d) & (df["J"] == J)]
                row1 = {"p": p, "a": a, "delta": d, "J": J}
                row2 = {"p": p, "a": a, "delta": d, "J": J}
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


def run_timing(cfg, ckpt_path):
    import warnings
    warnings.filterwarnings("ignore")
    dnn = load_surrogate(ckpt_path)
    df = pd.read_csv(DATA)
    tr = df[df["rep"].isin([1, 2, 3, 4, 5, 6, 7, 8])]
    X = np.column_stack([np.log10(tr["p"].to_numpy()), tr["a"].to_numpy(), tr["delta"].to_numpy()])
    gp = fit_gp_surrogate_3d(X, np.log10(tr["d_bar"].to_numpy()), budget=cfg["gp_budget"])
    n_time = cfg["timing_iters"]
    rows = []
    for p in cfg["p_grid"]:
        for (a, d) in cfg["regimes"]:
            J = cfg["J_grid"][-1]
            rng = np.random.default_rng(0)
            tp = solve_tp(1, a, p, 20)
            obs = summary_stat(*fluc_exp(1, a, d, p, tp, J, rng, use_slow=True))
            th0 = float(np.log10(p))
            tp_fn = lambda th: solve_tp(1, a, 10.0 ** th, 20)
            sk = dict(Z0=1, J=J, tp_fn=tp_fn, use_slow=True)
            per = {"p": p, "a": a, "delta": d, "J": J}
            for name, kw in [("ABC-MCMC", dict(backend="sim", sim_kwargs=sk, ns=cfg["ns"])),
                             ("GPS-ABC", dict(backend="gp", surrogate=gp)),
                             ("DNN-ABC", dict(backend="dnn", surrogate=dnn))]:
                t = time.time()
                run_abc_mcmc(obs, a_known=a, delta_known=d, n_mcmc=n_time, theta_init=th0,
                             eps=cfg["eps"], rng=np.random.default_rng(1),
                             prior_range=PRIOR_RANGE, **kw)
                per[name] = (time.time() - t) / n_time * 100.0
            rows.append(per)
            print(f"  timed p={p:.0e} a={a} d={d} J={J}: ABC-MCMC={per['ABC-MCMC']:.2f}s "
                  f"GPS-ABC={per['GPS-ABC']:.3f}s DNN-ABC={per['DNN-ABC']:.3f}s per 100 iter")
    return pd.DataFrame(rows)


def _reg(r):
    return f"a={r['a']:.1f},d={r['delta']:.1f}"


def fmt_table1(t1, methods=("MOM", "MLE", "ABC-MCMC", "GPS-ABC", "DNN-ABC")):
    lines = ["| p | regime | J | " + " | ".join(methods) + " |",
             "|---|---|---|" + "|".join(["---"] * len(methods)) + "|"]
    for _, r in t1.iterrows():
        cells = [f"{r[m]:.2e} ({r[m+'_nrmse']:.2f})" for m in methods]
        lines.append(f"| {r['p']:.0e} | {_reg(r)} | {int(r['J'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_table2(t2, methods=("ABC-MCMC", "GPS-ABC", "DNN-ABC")):
    lines = ["| p | regime | J | " + " | ".join(methods) + " |",
             "|---|---|---|" + "|".join(["---"] * len(methods)) + "|"]
    for _, r in t2.iterrows():
        cells = [f"{r[m]:.2e}" if m in r and np.isfinite(r[m]) else "-" for m in methods]
        lines.append(f"| {r['p']:.0e} | {_reg(r)} | {int(r['J'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_table3(t3):
    methods = ["ABC-MCMC", "GPS-ABC", "DNN-ABC"]
    lines = ["| p | regime | J | " + " | ".join(m + " (s/100it)" for m in methods) + " | ABC/DNN speedup |",
             "|---|---|---|---|---|---|---|"]
    for _, r in t3.iterrows():
        speed = r["ABC-MCMC"] / r["DNN-ABC"] if r["DNN-ABC"] > 0 else np.nan
        lines.append(f"| {r['p']:.0e} | {_reg(r)} | {int(r['J'])} | {r['ABC-MCMC']:.2f} | "
                     f"{r['GPS-ABC']:.3f} | {r['DNN-ABC']:.3f} | {speed:.0f}x |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--nmcmc", type=int, default=600)
    ap.add_argument("--burnin", type=int, default=250)
    ap.add_argument("--ns", type=int, default=6, help="sims per ABC-MCMC iteration")
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--timing-iters", type=int, default=40)
    ap.add_argument("--gp-budget", type=int, default=GP_BUDGET)
    ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 2) - 2))
    ap.add_argument("--p-grid", type=float, nargs="+", default=[1e-3, 1e-2])
    ap.add_argument("--J-grid", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--retrain", action="store_true", help="retrain the DNN surrogate first")
    args = ap.parse_args()

    ckpt = MODEL_DIR / "surrogate_3d.pt"
    if args.retrain or not ckpt.exists():
        print("training DNN surrogate...")
        train_run(str(DATA))

    cfg = dict(reps=args.reps, nmcmc=args.nmcmc, burnin=args.burnin, ns=args.ns,
               eps=args.eps, timing_iters=args.timing_iters, workers=args.workers,
               gp_budget=args.gp_budget, p_grid=list(args.p_grid),
               J_grid=list(args.J_grid), regimes=REGIMES)
    with open(LOG_DIR / "experiment_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    t_start = time.time()
    print("\n=== Phase A: accuracy (Table 1 & 2) ===")
    raw = run_accuracy(cfg, str(ckpt))
    raw.to_csv(LOG_DIR / "raw_replicates.csv", index=False)
    t1, t2 = aggregate_tables(raw, cfg)
    t1.to_csv(TABLE_DIR / "table1_mse.csv", index=False)
    t2.to_csv(TABLE_DIR / "table2_cilength.csv", index=False)

    print("\n=== Phase B: timing (Table 3) ===")
    t3 = run_timing(cfg, str(ckpt))
    t3.to_csv(TABLE_DIR / "table3_timing.csv", index=False)

    tbl1_md, tbl2_md, tbl3_md = fmt_table1(t1), fmt_table2(t2), fmt_table3(t3)
    with open(TABLE_DIR / "TABLES.md", "w") as f:
        f.write("# Reproduced tables with DNN-ABC column (3-D regime)\n\n")
        f.write(f"Config: {json.dumps(cfg)}\n\n")
        f.write("## Table 1 - MSE of p-hat, and (sqrt(MSE)/p) in parentheses\n\n")
        f.write(tbl1_md + "\n\n")
        f.write("## Table 2 - mean 95% interval length of p-hat\n\n")
        f.write(tbl2_md + "\n\n")
        f.write("## Table 3 - seconds per 100 MCMC iterations\n\n")
        f.write(tbl3_md + "\n")
    print("\n" + tbl1_md + "\n\n" + tbl3_md)
    print(f"\ntotal wall time: {(time.time()-t_start)/60:.1f} min")
    print(f"results written to {RESULTS}")


if __name__ == "__main__":
    main()
