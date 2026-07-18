"""Reproduce Table 1 (MSE), Table 2 (95% interval length) and Table 3 (compute
time) from Lu, Zhu & Wu (2023) -- adding a DNN-ABC column alongside the paper's
MOM/MLE, ABC-MCMC and GPS-ABC columns.

Experimental design (constant-mutation-rate, 1-D):
  truth  = exact/slow simulator (Algorithm 2), matching the DNN's training data.
  grid   = p in {1e-4, 1e-3, 1e-2}, J in {10, 50, 100}  (paper's slow-sim regime).
  For each (p, J) and each replicate:
     - simulate observed fluctuation data at p (slow sim), form obs = mean sqrt(X/Z),
     - estimate p with: MOM, MLE, ABC-MCMC (slow sim inside the loop),
       GPS-ABC (GP surrogate), DNN-ABC (our heteroscedastic MLP surrogate).
  Aggregate over replicates:
     Table 1 -> MSE and sqrt(MSE)/p ; Table 2 -> mean 95% interval length.
  Table 3 (timing) is a separate single-process pass timing each method's
  seconds / 100 MCMC iterations.

All ABC-MCMC methods share the same sampler, prior, proposal and eps, so the
only thing that changes across the ABC columns is how the summary statistic
(and its uncertainty) is obtained: brute-force simulation vs. GP vs. DNN.

Usage (defaults are a feasible local scale; crank up for the paper's scale):
    python run_experiments.py --reps 8 --nmcmc 600 --burnin 250 --ns 8
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
from train import load_surrogate, run as train_run
from paths import DATA, RESULTS, TABLE_DIR, FIG_DIR, MODEL_DIR, LOG_DIR
# Prior bounded to [-5,-2]: keeps the ABC-MCMC baseline's *slow* simulator calls
# feasible. Extending toward p=1e-8 lets chains wander into the exponential-cost
# region (tp ~ log(1/p), population ~ e^{a*tp}), where one slow-sim call can take
# minutes. The paper likewise uses a bounded prior ([-5,-1]) for the slow regime;
# the surrogates are still trained on the full [-8,-2] and only queried in-range.
PRIOR_RANGE = (-5.0, -2.0)

# worker globals (populated by _init_worker)
_G = {}


def _init_worker(ckpt_path, data_path, cfg):
    import warnings
    warnings.filterwarnings("ignore")
    dnn = load_surrogate(ckpt_path)
    df = pd.read_csv(data_path)
    tr = df[df["rep"].isin([1, 2, 3, 4, 5, 6, 7, 8])]
    x = np.log10(tr["p"].to_numpy())
    y = np.log10(tr["d_bar"].to_numpy())
    gp = fit_gp_surrogate(x, y, budget=None)
    _G["dnn"], _G["gp"], _G["cfg"] = dnn, gp, cfg


def _clamp_init(p_hat):
    lo, hi = PRIOR_RANGE
    th = np.log10(max(p_hat, 1e-8)) if (p_hat is not None and np.isfinite(p_hat) and p_hat > 0) else 0.5 * (lo + hi)
    return float(min(max(th, lo + 1e-6), hi - 1e-6))


def _one_replicate(task):
    p_true, J, rep = task
    cfg = _G["cfg"]
    dnn, gp = _G["dnn"], _G["gp"]
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
        out[name + "_acc"] = acc
    return out


def run_accuracy(cfg, ckpt_path):
    tasks = [(p, J, r) for p in cfg["p_grid"] for J in cfg["J_grid"]
             for r in range(cfg["reps"])]
    n = len(tasks)
    print(f"accuracy: {n} tasks ({len(cfg['p_grid'])}p x {len(cfg['J_grid'])}J x {cfg['reps']} reps) "
          f"on {cfg['workers']} workers", flush=True)
    rows = []
    t0 = time.time()
    with Pool(cfg["workers"], initializer=_init_worker,
              initargs=(ckpt_path, str(DATA), cfg)) as pool:
        # imap_unordered so we can log real progress + ETA (pool.map is opaque)
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


def run_timing(cfg, ckpt_path):
    """Single-process seconds / 100 MCMC iterations for each ABC method."""
    import warnings
    warnings.filterwarnings("ignore")
    dnn = load_surrogate(ckpt_path)
    df = pd.read_csv(DATA)
    tr = df[df["rep"].isin([1, 2, 3, 4, 5, 6, 7, 8])]
    gp = fit_gp_surrogate(np.log10(tr["p"].to_numpy()),
                          np.log10(tr["d_bar"].to_numpy()), budget=None)
    n_time = cfg["timing_iters"]
    rows = []
    for p in cfg["p_grid"]:
        for J in cfg["J_grid"]:
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
    if args.retrain or not ckpt.exists():
        print("training DNN surrogate...")
        train_run(str(DATA))

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

    print("\n=== Phase B: timing (Table 3) ===")
    t3 = run_timing(cfg, str(ckpt))
    t3.to_csv(TABLE_DIR / "table3_timing.csv", index=False)
    timing_plot(t3, FIG_DIR / "table3_timing.png")

    tbl1_md = fmt_table1(t1)
    tbl2_md = fmt_table2(t2)
    tbl3_md = fmt_table3(t3)
    with open(TABLE_DIR / "TABLES.md", "w") as f:
        f.write("# Reproduced tables with DNN-ABC column\n\n")
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
