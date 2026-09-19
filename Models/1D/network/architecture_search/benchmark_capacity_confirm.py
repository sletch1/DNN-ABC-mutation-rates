"""Does architecture size matter downstream, at the scale Table 1 actually uses?

benchmark_capacity.py answers the curve-fit half of this question (does the
response function itself need a 128-64 network?). This answers the harder,
more expensive half properly: run every candidate size through the ACTUAL
ABC-MCMC comparison the surrogate is used for, at the same replicate count and
Monte Carlo rigor as Table 1 in the paper, rather than a cheaper stand-in.

Two stages, named for and matching run_all.py's own vocabulary exactly:
  STAGE 1 (quick):  --quick settings (2 replicates, one (p,J) cell) -- a cheap
                     screen to catch anything catastrophically broken before
                     paying for the expensive stage.
  STAGE 2 (full):   the paper's own Table 1 settings (40 replicates, the full
                     p x J grid, 600 MCMC iterations) for every candidate that
                     survives the screen, with Monte Carlo standard errors
                     attached to every comparison exactly as Section on
                     simulation design and reporting does throughout the paper
                     (Delta/SE against the deployed 128-64, not just a nominal
                     ranking).

The exact-simulator ABC-MCMC baseline is excluded throughout: its behaviour
cannot depend on the surrogate's architecture, so recomputing it per candidate
would only add runtime, not information. GPS-ABC is included at both stages as
the fixed reference point.

Run: python benchmark_capacity_confirm.py   (writes results/logs/benchmark_capacity_confirm.md)
"""
import sys, time, warnings
from pathlib import Path
from multiprocessing import Pool

warnings.filterwarnings("ignore")

ROOT = Path("/Users/sachinletchumanan/Desktop/Research/Research-Code/DNN_ABC/Models/1D")
for d in (ROOT, ROOT / "network", ROOT / "network" / "architecture_search", ROOT / "abc"):
    sys.path.insert(0, str(d))

import numpy as np
import pandas as pd

from benchmark_arch import load, train_one, curve_mse
from surrogates import fit_gp_surrogate
from simulator import solve_tp, fluc_exp, summary_stat
from abc_mcmc import run_abc_mcmc, point_and_interval
from paths import DATA, LOG_DIR

PRIOR_RANGE = (-5.0, -2.0)
EPS = 0.005

QUICK = dict(reps=2,  nmcmc=120, burnin=40,  p_grid=[1e-2],
            J_grid=[10],           workers=6)
FULL  = dict(reps=40, nmcmc=600, burnin=250, p_grid=[1e-4, 1e-3, 1e-2],
            J_grid=[10, 50, 100],  workers=10)

# The six sizes from the pasted table -- the deployed network down to a
# single 8-unit layer. No linear control here; that candidate was already an
# unambiguous loser in the prior capacity sweep and isn't worth re-testing.
CANDIDATES = {
    "128-64 [deployed]": (128, 64),
    "64-32":             (64, 32),
    "32-16":             (32, 16),
    "16-8":              (16, 8),
    "16 (1 layer)":      (16,),
    "8 (1 layer)":       (8,),
}


def n_params(hidden_dims, in_dim=1):
    dims = [in_dim] + list(hidden_dims)
    n = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    n += 2 * (dims[-1] + 1)
    return n


_G = {}
def _init_worker(dnn, gp, nmcmc, burnin):
    _G["dnn"], _G["gp"], _G["nmcmc"], _G["burnin"] = dnn, gp, nmcmc, burnin

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
    for name, kw, nmcmc_, burnin_ in [
        ("GPS-ABC", dict(backend="gp", surrogate=gp), _G["nmcmc"], _G["burnin"]),
        ("DNN-ABC", dict(backend="dnn", surrogate=dnn), _G["nmcmc"], _G["burnin"]),
    ]:
        samples, acc = run_abc_mcmc(obs, n_mcmc=nmcmc_, theta_init=th0, s=0.15,
                                    rng=np.random.default_rng(rng.integers(2**63 - 1)),
                                    prior_range=PRIOR_RANGE, lam=2.0, eps=EPS, **kw)
        p_hat, ci_lo, ci_hi, ci_len = point_and_interval(samples, burnin_)
        out[name] = p_hat
        out[name + "_cilen"] = ci_len
        out[name + "_acc"] = acc
    return out


def run_grid(surr, gp, cfg):
    tasks = [(p, J, r) for p in cfg["p_grid"] for J in cfg["J_grid"] for r in range(cfg["reps"])]
    with Pool(cfg["workers"], initializer=_init_worker,
              initargs=(surr, gp, cfg["nmcmc"], cfg["burnin"])) as pool:
        rows = list(pool.imap_unordered(_one_replicate, tasks))
    df = pd.DataFrame(rows)
    per_cell = []
    for p in cfg["p_grid"]:
        for J in cfg["J_grid"]:
            sub = df[(df["p_true"] == p) & (df["J"] == J)]
            for name in ("GPS-ABC", "DNN-ABC"):
                est = sub[name].to_numpy(dtype=float); est = est[np.isfinite(est)]
                mse = float(np.mean((est - p) ** 2)) if len(est) else np.nan
                mcse = float(np.std((est - p) ** 2, ddof=1) / np.sqrt(len(est))) if len(est) > 1 else np.nan
                cl = sub[name + "_cilen"].to_numpy(dtype=float); cl = cl[np.isfinite(cl)]
                per_cell.append(dict(p=p, J=J, method=name, mse=mse, mcse=mcse,
                                     cilen=float(np.mean(cl)) if len(cl) else np.nan,
                                     n=len(est)))
    cells = pd.DataFrame(per_cell)
    dnn = cells[cells.method == "DNN-ABC"]
    gpc = cells[cells.method == "GPS-ABC"]
    return dict(mean_mse=dnn.mse.mean(), mean_cilen=dnn.cilen.mean(),
               gp_mean_mse=gpc.mse.mean(), gp_mean_cilen=gpc.cilen.mean(),
               cells=cells)


def main():
    (x_tr, y_tr), (x_va, y_va), (x_true, y_true) = load()
    tr8 = pd.read_csv(DATA); tr8 = tr8[tr8["rep"].isin([1,2,3,4,5,6,7,8])]
    gp = fit_gp_surrogate(np.log10(tr8["p"].to_numpy()), np.log10(tr8["d_bar"].to_numpy()), budget=None)

    # ---- STAGE 1: quick screen -------------------------------------------
    print("="*90); print("STAGE 1 -- QUICK SCREEN (run_all.py --quick settings: "
                          "reps=2, 1 cell p=1e-2 J=10)"); print("="*90, flush=True)
    surrs = {}
    quick_rows = []
    for name, hidden in CANDIDATES.items():
        surr = train_one(dict(hidden_dims=hidden, activation="gelu", use_bn=False, dropout=0.0),
                         x_tr, y_tr, x_va, y_va, seed=0)
        surrs[name] = surr
        t0 = time.time()
        r = run_grid(surr, gp, QUICK)
        dt = time.time() - t0
        print(f"  [{name}] quick DNN MSE={r['mean_mse']:.3e}  CIlen={r['mean_cilen']:.3e}  "
              f"({dt:.1f}s)", flush=True)
        quick_rows.append(dict(name=name, params=n_params(hidden),
                               quick_mse=r["mean_mse"], quick_cilen=r["mean_cilen"]))
    quick_df = pd.DataFrame(quick_rows).sort_values("quick_mse")
    print("\nQuick screen, sorted by MSE:")
    print(quick_df.to_string(index=False), flush=True)

    # Promising = not a clear, order-of-magnitude outlier on the quick screen.
    # With reps=2 at a single cell this is a coarse filter by design; anything
    # within ~2x of the best quick MSE moves on to the full confirm.
    best_quick = quick_df.quick_mse.min()
    promising = quick_df[quick_df.quick_mse <= 2.0 * best_quick]["name"].tolist()
    print(f"\nPromising (within 2x of best quick MSE {best_quick:.3e}): {promising}\n", flush=True)

    # ---- STAGE 2: full paper-scale confirm --------------------------------
    print("="*90); print("STAGE 2 -- FULL CONFIRM (paper settings: reps=40, "
                          "full 3x3 grid, nmcmc=600, burnin=250)"); print("="*90, flush=True)
    full_rows = []
    for name in promising:
        hidden = CANDIDATES[name]
        t0 = time.time()
        r = run_grid(surrs[name], gp, FULL)
        dt = time.time() - t0
        print(f"  [{name}] FULL DNN MSE={r['mean_mse']:.4e}  CIlen={r['mean_cilen']:.4e}  "
              f"(GP: MSE={r['gp_mean_mse']:.4e} CIlen={r['gp_mean_cilen']:.4e})  "
              f"[{dt/60:.1f}m]", flush=True)
        full_rows.append(dict(name=name, params=n_params(hidden),
                              full_mse=r["mean_mse"], full_cilen=r["mean_cilen"],
                              gp_mse=r["gp_mean_mse"], gp_cilen=r["gp_mean_cilen"],
                              cells=r["cells"]))

    full_df = pd.DataFrame(full_rows).sort_values("full_mse")
    print("\n" + "="*90)
    print("FULL-SCALE RESULTS (sorted by downstream MSE)")
    print("="*90)
    print(full_df.drop(columns="cells").to_string(index=False))

    # Delta/SE of every candidate against the deployed network, per cell,
    # exactly as Table 1 in the manuscript reports it -- a nominal ranking
    # alone can't say whether a gap is real at this replicate count.
    deployed = "128-64 [deployed]"

    def dnn_only(cells):
        return cells[cells.method == "DNN-ABC"].set_index(["p", "J"])

    dep_dnn = dnn_only(full_df.loc[full_df.name == deployed, "cells"].iloc[0])
    print("\nDelta/SE against the deployed network (Table-1-style; |.|>=2 is resolved):")
    resolved_summary = []
    for _, row in full_df.iterrows():
        if row["name"] == deployed:
            continue
        cand_cells = dnn_only(row["cells"])
        ts = []
        for idx in dep_dnn.index:
            se = float(np.sqrt(cand_cells.loc[idx, "mcse"] ** 2 + dep_dnn.loc[idx, "mcse"] ** 2))
            d = float(dep_dnn.loc[idx, "mse"] - cand_cells.loc[idx, "mse"])  # + => candidate better
            ts.append(d / se if se > 0 else float("nan"))
        n_resolved = sum(1 for t_ in ts if abs(t_) >= 2)
        max_t = max(abs(t_) for t_ in ts)
        print(f"  {row['name']:<20} resolved in {n_resolved}/9 cells, max |Delta/SE| = {max_t:.2f}")
        resolved_summary.append((row["name"], n_resolved, max_t))

    out = full_df.drop(columns="cells")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Capacity confirm: downstream accuracy/precision at Table-1 scale\n",
             "Quick screen (run_all.py --quick settings) followed by a full paper-scale "
             f"confirm (40 reps, full p x J grid) for every candidate. GP baseline: "
             f"MSE={full_df.iloc[0]['gp_mse']:.4e}, 95% CI={full_df.iloc[0]['gp_cilen']:.4e}.\n",
             "| Architecture | Parameters | Downstream MSE | Downstream 95% CI | "
             "Resolved vs. deployed | max \\|Delta/SE\\| |",
             "|---|---|---|---|---|---|"]
    resolved_map = {n: (r, m) for n, r, m in resolved_summary}
    for _, row in out.iterrows():
        r_m = resolved_map.get(row["name"], ("--", 0.0))
        lines.append(f"| {row['name']} | {row['params']} | {row['full_mse']:.4e} | "
                     f"{row['full_cilen']:.4e} | {r_m[0]}/9 | {r_m[1]:.2f} |")
    (LOG_DIR / "benchmark_capacity_confirm.md").write_text("\n".join(lines) + "\n")
    print("\nwritten -> results/logs/benchmark_capacity_confirm.md")


if __name__ == "__main__":
    main()
