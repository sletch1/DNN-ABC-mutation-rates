"""Does the deployed 128-64 network need to be that large?

The width/depth of Table 1 (benchmark_arch.py) was never itself swept -- every
candidate there differs in activation and BatchNorm, not in size below 128-64.
This asks the orthogonal question directly: holding activation (GELU) and the
no-BatchNorm choice fixed, how far can the network shrink before mean-curve fit
quality actually degrades, and does that degradation survive into the
downstream ABC-MCMC comparison the surrogate is actually used for?

Two things are measured for each candidate width, mirroring the two capacity
diagnostics used for the two-stage model in Study II (README/manuscript
Section on why the 3-D surrogate is small):
  1. Mean-curve MSE against the GP baseline, averaged over 3 random seeds
     (the same convention as the activation ablation in benchmark_arch.py's
     docstring), reported as mean +/- sd across seeds.
  2. Downstream ABC-MCMC accuracy (MSE of p-hat) and precision (95% credible
     interval length) on the paper's p x J grid, using one representative
     seed per candidate. The expensive EXACT-simulator ABC-MCMC baseline is
     skipped here: its behaviour cannot depend on the surrogate's size, so
     recomputing it for every candidate would only add runtime, not signal.

Run: python benchmark_capacity.py   (writes results/logs/benchmark_capacity.md)
"""
import sys
import time
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

# --- make sibling code folders + paths.py importable (package uses flat imports) ---
_ROOT = Path(__file__).resolve().parents[2]
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc", _ROOT / "figures"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd

from benchmark_arch import load, train_one, curve_mse
from surrogates import DNNSurrogate, fit_gp_surrogate
from simulator import solve_tp, fluc_exp, summary_stat
from abc_mcmc import run_abc_mcmc, point_and_interval
from paths import DATA, LOG_DIR

warnings.filterwarnings("ignore")

SEEDS = (0, 1, 2)
PRIOR_RANGE = (-5.0, -2.0)
REPS = 20                       # per (p, J) cell for the downstream check
NMCMC, BURNIN, EPS = 600, 250, 0.005
P_GRID = [1e-4, 1e-3, 1e-2]
J_GRID = [10, 50, 100]
WORKERS = 10

# The deployed size plus a ladder shrinking it in both width and depth, down to
# a linear control (zero hidden layers) so the comparison brackets "no network
# at all" the same way Table capacity3D does for the two-stage model.
CANDIDATES = {
    "128-64 [deployed]": (128, 64),
    "64-32":             (64, 32),
    "32-16":             (32, 16),
    "16-8":              (16, 8),
    "16 (1 layer)":      (16,),
    "8 (1 layer)":       (8,),
    "Linear (control)":  (),
}


def n_params(hidden_dims, in_dim=1):
    dims = [in_dim] + list(hidden_dims)
    n = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))  # trunk
    n += 2 * (dims[-1] + 1)  # mean head + logvar head
    return n


_G = {}


def _init_worker(dnn, gp):
    _G["dnn"], _G["gp"] = dnn, gp


def _one_replicate(task):
    p_true, J, rep = task
    dnn, gp = _G["dnn"], _G["gp"]
    rng = np.random.default_rng(10_000 * int(round(-np.log10(p_true))) + 100 * J + rep)
    tp = solve_tp(1, 1, p_true, 20)
    Z_vec, X_vec = fluc_exp(1, 1, 1, p_true, tp, J, rng, use_slow=True)
    obs = summary_stat(Z_vec, X_vec)
    th0 = float(np.clip(np.log10(p_true) + rng.normal(0, 0.3),
                        PRIOR_RANGE[0] + 1e-6, PRIOR_RANGE[1] - 1e-6))
    out = {"p_true": p_true, "J": J, "rep": rep}
    for name, kw in [("GPS-ABC", dict(backend="gp", surrogate=gp)),
                     ("DNN-ABC", dict(backend="dnn", surrogate=dnn))]:
        samples, acc = run_abc_mcmc(obs, n_mcmc=NMCMC, theta_init=th0, s=0.15,
                                    rng=np.random.default_rng(rng.integers(2**63 - 1)),
                                    prior_range=PRIOR_RANGE, lam=2.0, eps=EPS, **kw)
        p_hat, ci_lo, ci_hi, ci_len = point_and_interval(samples, BURNIN)
        out[name] = p_hat
        out[name + "_cilen"] = ci_len
    return out


def downstream_check(surr, gp):
    """Mean MSE and 95% CI length for one surrogate, over the full p x J grid."""
    tasks = [(p, J, r) for p in P_GRID for J in J_GRID for r in range(REPS)]
    with Pool(WORKERS, initializer=_init_worker, initargs=(surr, gp)) as pool:
        rows = list(pool.imap_unordered(_one_replicate, tasks))
    df = pd.DataFrame(rows)
    cell_mse = []
    for p in P_GRID:
        for J in J_GRID:
            sub = df[(df["p_true"] == p) & (df["J"] == J)]
            est = sub["DNN-ABC"].to_numpy(dtype=float)
            est = est[np.isfinite(est)]
            cell_mse.append(float(np.mean((est - p) ** 2)))
    cilen = df["DNN-ABC_cilen"].to_numpy(dtype=float)
    return float(np.mean(cell_mse)), float(np.nanmean(cilen))


def main():
    (x_tr, y_tr), (x_va, y_va), (x_true, y_true) = load()
    print(f"train n={len(x_tr)}  seeds={SEEDS}  downstream reps/cell={REPS}\n", flush=True)

    gp = fit_gp_surrogate(x_tr, y_tr, budget=None)
    gp_curve_mse = curve_mse(gp.predict, x_true, y_true)
    print(f"GP mean-curve MSE = {gp_curve_mse:.5e}", flush=True)
    gp_mse, gp_cilen = downstream_check(gp, gp)
    print(f"GP downstream: MSE={gp_mse:.4e}  95% CI len={gp_cilen:.4e}\n", flush=True)

    rows = []
    for name, hidden in CANDIDATES.items():
        curve_mses = []
        seed_surrs = []
        for seed in SEEDS:
            surr = train_one(dict(hidden_dims=hidden, activation="gelu",
                                  use_bn=False, dropout=0.0),
                             x_tr, y_tr, x_va, y_va, seed=seed)
            curve_mses.append(curve_mse(surr.predict, x_true, y_true))
            seed_surrs.append(surr)
        mean_mse, sd_mse = float(np.mean(curve_mses)), float(np.std(curve_mses, ddof=1))

        # Representative seed for the downstream check: whichever seed's
        # curve fit is closest to the across-seed mean, so the downstream
        # number is not cherry-picked from the best or worst run.
        rep_idx = int(np.argmin(np.abs(np.array(curve_mses) - mean_mse)))
        t0 = time.time()
        abc_mse, abc_cilen = downstream_check(seed_surrs[rep_idx], gp)
        abc_time = time.time() - t0

        params = n_params(hidden)
        vs_gp_pct = (mean_mse / gp_curve_mse - 1) * 100
        print(f"[{name}] params={params}  curve-MSE={mean_mse:.5e} +/- {sd_mse:.1e} "
              f"({vs_gp_pct:+.1f}% vs GP, {len(SEEDS)} seeds)  "
              f"downstream MSE={abc_mse:.4e} CIlen={abc_cilen:.4e}  "
              f"[{abc_time/60:.1f}m]", flush=True)

        rows.append(dict(name=name, hidden_dims=str(hidden), params=params,
                         curve_mse=mean_mse, curve_mse_sd=sd_mse,
                         curve_vs_gp_pct=vs_gp_pct,
                         abc_mse=abc_mse, abc_cilen=abc_cilen))

    out = pd.DataFrame(rows)
    lines = ["| Hidden layers | Parameters | Curve MSE (3-seed mean +/- sd) | vs GP | "
             "Downstream MSE | Downstream 95% CI length |",
             "|---|---|---|---|---|---|"]
    lines.append(f"| GP (GPS-ABC baseline) | -- | {gp_curve_mse:.3e} | --- | "
                 f"{gp_mse:.3e} | {gp_cilen:.3e} |")
    for _, r in out.iterrows():
        lines.append(f"| {r['name']} | {r['params']} | "
                     f"{r['curve_mse']:.3e} +/- {r['curve_mse_sd']:.1e} | "
                     f"{r['curve_vs_gp_pct']:+.1f}% | {r['abc_mse']:.3e} | {r['abc_cilen']:.3e} |")
    table = "\n".join(lines)
    print("\n" + table)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "benchmark_capacity.md").write_text(
        "# Capacity benchmark: how small can the 1-D surrogate be?\n\n"
        f"GP mean-curve MSE = {gp_curve_mse:.5e}. Curve MSE is against the "
        f"denoised response curve, averaged over {len(SEEDS)} seeds. Downstream "
        f"MSE/CI length are DNN-ABC on the full p x J grid, {REPS} reps/cell, "
        "one representative seed per candidate (see script docstring); the "
        "exact-simulator ABC-MCMC baseline is not re-run per candidate since it "
        "does not depend on the surrogate.\n\n" + table + "\n")
    print("\nwritten -> results/logs/benchmark_capacity.md")


if __name__ == "__main__":
    main()
