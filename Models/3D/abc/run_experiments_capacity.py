"""Does 3-D surrogate size matter downstream, at the scale Table 1 actually uses?

benchmark_round2.py (network/architecture_search/) already answers the
CURVE-FIT half of this question for this package -- it walked the same
width/depth ladder down from 256-128-64 to a linear control and reported
mean-curve MSE against the irreducible noise floor, which is where the
deployed 64-32 shape comes from. It never re-ran the actual downstream
ABC-MCMC comparison for the smaller candidates, so it could not say whether a
curve-fit difference (or the lack of one) survives into the task the
surrogate is actually used for. This closes that gap, mirroring exactly the
two-stage design applied to the 1-D package's own capacity question
(Models/1D/network/architecture_search/benchmark_capacity_confirm.py):

  STAGE 1 (quick):  this project's own --quick settings (reps=2, nmcmc=300,
                     burnin=100) across all 3 truth triples -- a cheap screen
                     to catch anything catastrophically broken.
  STAGE 2 (full):   the paper's own Table 1 settings (reps=16, nmcmc=3000,
                     burnin=1000), for every candidate that survives the
                     screen, with Monte Carlo standard errors on rmse_log
                     (mcse.py's own mcse_rmse, the same SE the paper's mcse.md
                     uses) attached to every comparison against the deployed
                     64-32.

The exact-simulator ABC-MCMC baseline is excluded throughout, as it already
is by default in this package's paper-scale run: its behaviour cannot depend
on the surrogate's architecture, so recomputing it per candidate would only
add runtime, not information. GPS-ABC is fit once and shared across every
candidate, as the fixed reference point.

Run: python run_experiments_capacity.py
Writes: results/logs/benchmark_capacity_confirm.md
"""
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search", _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from train import load_splits, train_model, calibrate_conformal, summary_from_cultures
from surrogates import DNNSurrogate3D, fit_gp_surrogate_3d
from simulator import fluc_exp_2stage, summary_stat
from abc_mcmc import run_abc_mcmc, summarize, ess, DEFAULT_BOX, DEFAULT_STEPS
from mcse import mcse_rmse
from paths import DATA, LOG_DIR

TRUTHS = [
    (1e-4, 1e-2, 3.0),
    (1e-4, 1e-2, 7.0),
    (2e-3, 8e-3, 5.0),
]
A, TP, Z0, J = 1.0, 10.0, 1, 100
MUT_TIME = "offspring"
EPS = 0.005
GP_BUDGET = 300

QUICK = dict(reps=2,  nmcmc=300,  burnin=100,  workers=6)
FULL  = dict(reps=16, nmcmc=3000, burnin=1000, workers=10)

# The same width/depth ladder as benchmark_round2.py, minus the linear
# control (already an unambiguous loser there -- not worth re-testing).
CANDIDATES = {
    "256-128-64":        (256, 128, 64),
    "128-64":            (128, 64),
    "64-32 [deployed]":  (64, 32),
    "32-16":             (32, 16),
    "16-8":              (16, 8),
    "8-4":               (8, 4),
}


def n_params(hidden, in_dim=3):
    dims = [in_dim] + list(hidden)
    n = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    n += 2 * (dims[-1] + 1)
    return n


_G = {}


def _init_worker(dnn, gp, nmcmc, burnin):
    import torch
    torch.set_num_threads(1)   # one process per replicate; don't oversubscribe
    _G.update(dnn=dnn, gp=gp, nmcmc=nmcmc, burnin=burnin)


def _one_replicate(task):
    p1, p2, tau, rep = task
    cfg_nmcmc, cfg_burnin = _G["nmcmc"], _G["burnin"]
    seed = abs(hash((round(p1, 12), round(p2, 12), round(tau, 3), J, rep))) % (2 ** 31)
    rng = np.random.default_rng(seed)
    Zv, Xv = fluc_exp_2stage(Z0, A, p1, p2, tau, TP, J, rng, use_slow=True, mut_time=MUT_TIME)
    obs = summary_stat(Zv, Xv)
    truth = dict(p1=p1, p2=p2, tau=tau)

    out = {"p1_true": p1, "p2_true": p2, "tau_true": tau, "rep": rep}
    for name, kw in [("GPS-ABC", dict(backend="gp", surrogate=_G["gp"])),
                     ("DNN-ABC", dict(backend="dnn", surrogate=_G["dnn"]))]:
        s, acc = run_abc_mcmc(obs, n_mcmc=cfg_nmcmc, steps=DEFAULT_STEPS, box=DEFAULT_BOX,
                              eps=EPS, rng=np.random.default_rng(rng.integers(2 ** 63 - 1)), **kw)
        post = summarize(s, cfg_burnin)
        for k in ("p1", "p2", "tau"):
            out[f"{name}_{k}"] = post[k]["mean"]
            out[f"{name}_{k}_cilen"] = post[k]["ci_len"]
            out[f"{name}_{k}_cov"] = int(post[k]["ci_lo"] <= truth[k] <= post[k]["ci_hi"])
    return out


def run_grid(surr, gp, cfg):
    tasks = [(p1, p2, tau, r) for (p1, p2, tau) in TRUTHS for r in range(cfg["reps"])]
    with Pool(cfg["workers"], initializer=_init_worker,
              initargs=(surr, gp, cfg["nmcmc"], cfg["burnin"])) as pool:
        rows = pd.DataFrame(list(pool.imap_unordered(_one_replicate, tasks)))

    per_param = []
    for (p1, p2, tau) in TRUTHS:
        sub = rows[(rows.p1_true == p1) & (rows.p2_true == p2) & (rows.tau_true == tau)]
        for method in ("GPS-ABC", "DNN-ABC"):
            for k, tv in (("p1", p1), ("p2", p2), ("tau", tau)):
                est = sub[f"{method}_{k}"].to_numpy(float)
                est = est[np.isfinite(est)]
                if k in ("p1", "p2"):
                    err = np.log10(np.maximum(est, 1e-300)) - np.log10(tv)
                else:
                    err = est - tv
                per_param.append(dict(p1=p1, p2=p2, tau=tau, method=method, param=k,
                                      rmse_log=float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
                                      mcse=mcse_rmse(err), n=len(err)))
    cells = pd.DataFrame(per_param)
    dnn = cells[cells.method == "DNN-ABC"]
    gpc = cells[cells.method == "GPS-ABC"]
    return dict(mean_rmse=dnn.rmse_log.mean(), gp_mean_rmse=gpc.rmse_log.mean(), cells=cells)


def main():
    (Xtr, ytr, _), (Xva, yva, _), _ = load_splits(DATA)
    print(f"train n={len(ytr)}  candidates={list(CANDIDATES)}\n", flush=True)

    gp = fit_gp_surrogate_3d(Xtr, ytr, budget=GP_BUDGET)
    print("GP baseline fit.\n", flush=True)

    # ---- STAGE 1: quick screen -------------------------------------------
    print("=" * 90); print("STAGE 1 -- QUICK SCREEN (reps=2, nmcmc=300, burnin=100, "
                           "all 3 truth triples)"); print("=" * 90, flush=True)
    surrs = {}
    quick_rows = []
    for name, hidden in CANDIDATES.items():
        arch = dict(kind="mlp", hidden=hidden, activation="gelu")
        model, xs, ys = train_model(Xtr, ytr, Xva, yva, arch=arch, seed=0)
        surr = DNNSurrogate3D(model, xs, ys, sd_scale=1.0, raw_inputs=False)
        surr.sd_scale = calibrate_conformal(surr, Xva, yva)
        surrs[name] = surr

        t0 = time.time()
        r = run_grid(surr, gp, QUICK)
        dt = time.time() - t0
        print(f"  [{name}] quick mean rmse_log={r['mean_rmse']:.3f}  ({dt:.1f}s)", flush=True)
        quick_rows.append(dict(name=name, params=n_params(hidden), quick_rmse=r["mean_rmse"]))

    quick_df = pd.DataFrame(quick_rows).sort_values("quick_rmse")
    print("\nQuick screen, sorted by mean rmse_log:")
    print(quick_df.to_string(index=False), flush=True)
    best_quick = quick_df.quick_rmse.min()
    promising = quick_df[quick_df.quick_rmse <= 1.5 * best_quick]["name"].tolist()
    print(f"\nPromising (within 1.5x of best quick rmse_log {best_quick:.3f}): {promising}\n",
          flush=True)

    # ---- STAGE 2: full paper-scale confirm --------------------------------
    print("=" * 90); print("STAGE 2 -- FULL CONFIRM (reps=16, nmcmc=3000, burnin=1000, "
                           "Table 1 scale)"); print("=" * 90, flush=True)
    full_rows = []
    for name in promising:
        t0 = time.time()
        r = run_grid(surrs[name], gp, FULL)
        dt = time.time() - t0
        print(f"  [{name}] FULL mean rmse_log={r['mean_rmse']:.4f}  "
              f"(GP: {r['gp_mean_rmse']:.4f})  [{dt/60:.1f}m]", flush=True)
        full_rows.append(dict(name=name, params=n_params(CANDIDATES[name]),
                              full_rmse=r["mean_rmse"], gp_rmse=r["gp_mean_rmse"],
                              cells=r["cells"]))

    full_df = pd.DataFrame(full_rows).sort_values("full_rmse")
    print("\n" + "=" * 90); print("FULL-SCALE RESULTS (sorted by mean rmse_log, lower=better)")
    print("=" * 90)
    print(full_df.drop(columns="cells").to_string(index=False))

    deployed = "64-32 [deployed]"
    dep_cells = full_df.loc[full_df.name == deployed, "cells"]
    resolved_summary = []

    def dnn_only(cells):
        return cells[cells.method == "DNN-ABC"].set_index(["p1", "p2", "tau", "param"])

    if len(dep_cells):
        dep = dnn_only(dep_cells.iloc[0])
        print("\nDelta/SE against the deployed network (per truth-triple x parameter cell, "
              "|.|>=2 is resolved):")
        for _, row in full_df.iterrows():
            if row["name"] == deployed:
                continue
            cand = dnn_only(row["cells"])
            ts = []
            for idx in dep.index:
                se = float(np.sqrt(cand.loc[idx, "mcse"] ** 2 + dep.loc[idx, "mcse"] ** 2))
                d = float(dep.loc[idx, "rmse_log"] - cand.loc[idx, "rmse_log"])  # + => candidate better
                ts.append(d / se if se > 0 else float("nan"))
            n_resolved = sum(1 for t_ in ts if abs(t_) >= 2)
            max_t = max(abs(t_) for t_ in ts)
            print(f"  {row['name']:<20} resolved in {n_resolved}/{len(ts)} cells, "
                  f"max |Delta/SE| = {max_t:.2f}")
            resolved_summary.append((row["name"], n_resolved, len(ts), max_t))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# 3-D capacity confirm: downstream accuracy at Table-1 scale\n",
             f"Quick screen (reps=2) followed by a full paper-scale confirm (reps=16, "
             f"Table 1's exact settings) for every candidate that survives it. "
             f"GP baseline mean rmse_log = {full_df.iloc[0]['gp_rmse']:.4f}.\n",
             "| Architecture | Parameters | Mean rmse_log (all params/truths) | "
             "Resolved vs. deployed | max \\|Delta/SE\\| |",
             "|---|---|---|---|---|"]
    resolved_map = {n: (r, t, m) for n, r, t, m in resolved_summary}
    for _, row in full_df.iterrows():
        r_m = resolved_map.get(row["name"], ("--", "--", 0.0))
        lines.append(f"| {row['name']} | {row['params']} | {row['full_rmse']:.4f} | "
                     f"{r_m[0]}/{r_m[1]} | {r_m[2]:.2f} |")
    (LOG_DIR / "benchmark_capacity_confirm.md").write_text("\n".join(lines) + "\n")
    print("\nwritten -> results/logs/benchmark_capacity_confirm.md")


if __name__ == "__main__":
    main()
