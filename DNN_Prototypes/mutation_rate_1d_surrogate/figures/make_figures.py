"""Generate the professor-facing figures for the README.

Uses only cheap operations (trained surrogate predictions, the pre-computed
raw_replicates.csv, and a few short surrogate-ABC chains), so it adds no
meaningful load. Produces:

  fig_uncertainty.png      - DNN heteroscedastic sd vs GP homoscedastic sd vs
                             the empirical replicate noise (target #5 evidence)
  fig_table1_mse.png       - nRMSE = sqrt(MSE)/p by method, one panel per p
  fig_posterior.png        - posterior densities of p from the three ABC methods
                             for one representative case, vs the truth
  fig_timing.png           - seconds/100 MCMC iterations (log scale), DNN vs rest

Run after run_experiments.py (needs results/surrogate_1d.pt, raw_replicates.csv,
table1_mse.csv, table3_timing.csv).
"""

import sys
import warnings
from pathlib import Path

# --- make sibling code folders + paths.py importable (package uses flat imports) ---
_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc", _ROOT / "figures"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulator import solve_tp, fluc_exp, summary_stat
from estimators import estimate_mom
from abc_mcmc import run_abc_mcmc, point_and_interval
from surrogates import fit_gp_surrogate
from train import load_surrogate
from paths import DATA, MODEL_DIR, TABLE_DIR, FIG_DIR

warnings.filterwarnings("ignore")

METHOD_COLORS = {"MOM": "#9e9e9e", "MLE": "#607d8b", "ABC-MCMC": "#e53935",
                 "GPS-ABC": "#43a047", "DNN-ABC": "#1e88e5"}


def _load():
    dnn = load_surrogate(MODEL_DIR / "surrogate_1d.pt")
    df = pd.read_csv(DATA)
    tr = df[df["rep"].isin([1, 2, 3, 4, 5, 6, 7, 8])]
    gp = fit_gp_surrogate(np.log10(tr["p"].to_numpy()),
                          np.log10(tr["d_bar"].to_numpy()), budget=None)
    return dnn, gp, df


def fig_uncertainty(dnn, gp, df):
    xg = np.linspace(-8, -2, 200)
    _, dnn_sd = dnn.predict(xg)
    _, gp_sd = gp.predict(xg)
    # empirical replicate std of log10(d_bar) per p grid point
    g = df.groupby("p")["d_bar"].apply(lambda s: np.std(np.log10(s), ddof=1))
    emp_x = np.log10(g.index.to_numpy())
    emp_y = g.to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(emp_x, emp_y, s=16, color="black", alpha=0.5,
               label="empirical replicate noise (data)")
    ax.plot(xg, dnn_sd, color=METHOD_COLORS["DNN-ABC"], lw=2.4,
            label="DNN predictive sd (heteroscedastic)")
    ax.plot(xg, gp_sd, color=METHOD_COLORS["GPS-ABC"], lw=2.4, ls="--",
            label="GP predictive sd (homoscedastic)")
    ax.set_xlabel("log10(p)")
    ax.set_ylabel("predictive std of log10(d_bar)")
    ax.set_title("Calibrated predictive uncertainty: DNN tracks the true\n"
                 "input-dependent noise; the GP's single noise term cannot")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_uncertainty.png", dpi=150)
    plt.close(fig)


def fig_table1(df_raw):
    t1 = pd.read_csv(TABLE_DIR / "table1_mse.csv")
    p_vals = sorted(t1["p"].unique())
    methods = ["MOM", "MLE", "ABC-MCMC", "GPS-ABC", "DNN-ABC"]
    fig, axes = plt.subplots(1, len(p_vals), figsize=(4.6 * len(p_vals), 4.2), sharey=True)
    if len(p_vals) == 1:
        axes = [axes]
    Js = sorted(t1["J"].unique())
    width = 0.16
    for ax, p in zip(axes, p_vals):
        sub = t1[t1["p"] == p].sort_values("J")
        xpos = np.arange(len(Js))
        for k, m in enumerate(methods):
            vals = [sub[sub["J"] == J][m + "_nrmse"].values[0] for J in Js]
            ax.bar(xpos + (k - 2) * width, vals, width, label=m,
                   color=METHOD_COLORS[m])
        ax.set_title(f"p = {p:.0e}")
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"J={J}" for J in Js])
        ax.set_xlabel("parallel cultures")
    axes[0].set_ylabel("nRMSE  =  sqrt(MSE) / p")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Table 1 visualized: estimation error by method "
                 "(lower is better; surrogates match ABC-MCMC)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_table1_mse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_posterior(dnn, gp, p_true=1e-3, J=100, n_mcmc=1500, burn=500):
    rng = np.random.default_rng(7)
    tp = solve_tp(1, 1, p_true, 20)
    Zv, Xv = fluc_exp(1, 1, 1, p_true, tp, J, rng, use_slow=True)
    obs = summary_stat(Zv, Xv)
    th0 = float(np.clip(np.log10(max(estimate_mom(Zv, Xv), 1e-8)), -7.99, -2.01))
    tp_fn = lambda th: solve_tp(1, 1, 10.0 ** th, 20)
    sk = dict(Z0=1, a=1, delta=1, J=J, tp_fn=tp_fn, use_slow=True, method="synthetic")

    runs = {
        "ABC-MCMC": dict(backend="sim", sim_kwargs=sk, ns=10),
        "GPS-ABC": dict(backend="gp", surrogate=gp),
        "DNN-ABC": dict(backend="dnn", surrogate=dnn),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, kw in runs.items():
        s, acc = run_abc_mcmc(obs, n_mcmc=n_mcmc, theta_init=th0, eps=0.005,
                              rng=np.random.default_rng(1), **kw)
        p_post = 10.0 ** s[burn:]
        ax.hist(p_post, bins=60, density=True, histtype="step", lw=2,
                color=METHOD_COLORS[name], label=f"{name} (acc {acc:.2f})")
    ax.axvline(p_true, color="black", ls="--", lw=1.5, label=f"true p = {p_true:.0e}")
    ax.set_xlabel("mutation probability p (posterior)")
    ax.set_ylabel("posterior density")
    ax.set_title(f"Posterior of p from the three ABC methods agree\n"
                 f"(single dataset, p={p_true:.0e}, J={J})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_posterior.png", dpi=150)
    plt.close(fig)


def fig_timing():
    t3 = pd.read_csv(TABLE_DIR / "table3_timing.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = [f"{r['p']:.0e}\nJ={int(r['J'])}" for _, r in t3.iterrows()]
    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, t3["ABC-MCMC"], w, label="ABC-MCMC", color=METHOD_COLORS["ABC-MCMC"])
    ax.bar(x, t3["GPS-ABC"], w, label="GPS-ABC (GP)", color=METHOD_COLORS["GPS-ABC"])
    ax.bar(x + w, t3["DNN-ABC"], w, label="DNN-ABC (ours)", color=METHOD_COLORS["DNN-ABC"])
    ax.set_yscale("log")
    ax.set_ylabel("seconds / 100 MCMC iterations (log scale)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Per-iteration cost: surrogate ABC is 10^2-10^3x faster than ABC-MCMC")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_timing.png", dpi=150)
    plt.close(fig)


def main():
    dnn, gp, df = _load()
    print("fig_uncertainty..."); fig_uncertainty(dnn, gp, df)
    print("fig_table1_mse..."); fig_table1(df)
    print("fig_timing..."); fig_timing()
    print("fig_posterior (runs a few short chains)..."); fig_posterior(dnn, gp)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
