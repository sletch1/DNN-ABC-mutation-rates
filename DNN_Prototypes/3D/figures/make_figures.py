"""Generate the professor-facing figures for the 3-D README.

Uses only cheap operations (trained surrogate predictions, the pre-computed
raw_replicates.csv, and a few short surrogate-ABC chains). Produces:

  fig_uncertainty.png  - DNN heteroscedastic sd vs GP (budget 300) homoscedastic
                         sd vs empirical replicate noise, along log10(p) at fixed
                         (a, delta) slices (target #5 evidence)
  fig_table1_mse.png   - nRMSE = sqrt(MSE)/p by method, one panel per (p, regime)
  fig_posterior.png    - posterior densities of p from the three ABC methods for
                         one representative 3-D case, vs the truth
  fig_timing.png       - seconds/100 MCMC iterations (log scale), DNN vs rest

Run after run_experiments.py (needs results/model/surrogate_3d.pt,
raw_replicates.csv, table1_mse.csv, table3_timing.csv).
"""

import sys
import warnings
from pathlib import Path

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
from surrogates import fit_gp_surrogate_3d
from train import load_surrogate
from paths import DATA, MODEL_DIR, TABLE_DIR, FIG_DIR

warnings.filterwarnings("ignore")

METHOD_COLORS = {"MOM": "#9e9e9e", "MLE": "#607d8b", "ABC-MCMC": "#e53935",
                 "GPS-ABC": "#43a047", "DNN-ABC": "#1e88e5"}


def _load():
    dnn = load_surrogate(MODEL_DIR / "surrogate_3d.pt")
    df = pd.read_csv(DATA)
    tr = df[df["rep"].isin([1, 2, 3, 4, 5, 6, 7, 8])]
    X = np.column_stack([np.log10(tr["p"].to_numpy()), tr["a"].to_numpy(), tr["delta"].to_numpy()])
    gp = fit_gp_surrogate_3d(X, np.log10(tr["d_bar"].to_numpy()), budget=300)
    return dnn, gp, df


def fig_uncertainty(dnn, gp, df):
    a_vals = np.sort(df["a"].unique()); d_vals = np.sort(df["delta"].unique())
    slices = [(a_vals[2], d_vals[2]), (a_vals[7], d_vals[7])]
    pg = np.linspace(-8, -2, 200)
    fig, axes = plt.subplots(1, len(slices), figsize=(6.6 * len(slices), 4.4), sharey=True)
    for ax, (a, d) in zip(axes, slices):
        sub = df[(np.isclose(df["a"], a)) & (np.isclose(df["delta"], d))]
        g = sub.groupby("p")["d_bar"].apply(lambda s: np.std(np.log10(s), ddof=1))
        Xg = np.column_stack([pg, np.full_like(pg, a), np.full_like(pg, d)])
        _, dnn_sd = dnn.predict(Xg)
        _, gp_sd = gp.predict(Xg)
        ax.scatter(np.log10(g.index.to_numpy()), g.to_numpy(), s=20, color="black",
                   alpha=0.55, label="empirical replicate noise")
        ax.plot(pg, dnn_sd, color=METHOD_COLORS["DNN-ABC"], lw=2.4,
                label="DNN predictive sd (heteroscedastic)")
        ax.plot(pg, gp_sd, color=METHOD_COLORS["GPS-ABC"], lw=2.4, ls="--",
                label="GP predictive sd (budget 300)")
        ax.set_title(f"a={a:.2f}, delta={d:.2f}", fontsize=10)
        ax.set_xlabel("log10(p)")
    axes[0].set_ylabel("predictive std of log10(d_bar)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Calibrated input-dependent uncertainty across the 3-D space:\n"
                 "the DNN's variance head tracks the true noise; the GP's is flatter", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_uncertainty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_table1():
    t1 = pd.read_csv(TABLE_DIR / "table1_mse.csv")
    t1["cell"] = t1.apply(lambda r: f"{r['p']:.0e}\na={r['a']:.1f},d={r['delta']:.1f}", axis=1)
    methods = ["MOM", "MLE", "ABC-MCMC", "GPS-ABC", "DNN-ABC"]
    Js = sorted(t1["J"].unique())
    fig, axes = plt.subplots(1, len(Js), figsize=(7.5 * len(Js), 4.6), sharey=True)
    if len(Js) == 1:
        axes = [axes]
    for ax, J in zip(axes, Js):
        sub = t1[t1["J"] == J].reset_index(drop=True)
        xpos = np.arange(len(sub)); width = 0.16
        for k, m in enumerate(methods):
            ax.bar(xpos + (k - 2) * width, sub[m + "_nrmse"], width, label=m,
                   color=METHOD_COLORS[m])
        ax.set_title(f"J = {J}")
        ax.set_xticks(xpos); ax.set_xticklabels(sub["cell"], fontsize=7)
        ax.set_xlabel("p and (a, delta) regime")
    axes[0].set_ylabel("nRMSE = sqrt(MSE) / p")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Table 1 visualized: estimation error by method across 3-D regimes "
                 "(lower is better)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_table1_mse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_posterior(dnn, gp, p_true=1e-2, a=1.5, delta=1.5, J=100, n_mcmc=1500, burn=500):
    rng = np.random.default_rng(7)
    tp = solve_tp(1, a, p_true, 20)
    Zv, Xv = fluc_exp(1, a, delta, p_true, tp, J, rng, use_slow=True)
    obs = summary_stat(Zv, Xv)
    th0 = float(np.clip(np.log10(max(estimate_mom(Zv, Xv), 1e-8)), -3.99, -1.51))
    tp_fn = lambda th: solve_tp(1, a, 10.0 ** th, 20)
    sk = dict(Z0=1, J=J, tp_fn=tp_fn, use_slow=True)
    runs = {
        "ABC-MCMC": dict(backend="sim", sim_kwargs=sk, ns=10),
        "GPS-ABC": dict(backend="gp", surrogate=gp),
        "DNN-ABC": dict(backend="dnn", surrogate=dnn),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, kw in runs.items():
        s, acc = run_abc_mcmc(obs, a_known=a, delta_known=delta, n_mcmc=n_mcmc,
                              theta_init=th0, eps=0.005, prior_range=(-4.0, -1.5),
                              rng=np.random.default_rng(1), **kw)
        p_post = 10.0 ** s[burn:]
        ax.hist(p_post, bins=60, density=True, histtype="step", lw=2,
                color=METHOD_COLORS[name], label=f"{name} (acc {acc:.2f})")
    ax.axvline(p_true, color="black", ls="--", lw=1.5, label=f"true p = {p_true:.0e}")
    ax.set_xlabel("mutation probability p (posterior)")
    ax.set_ylabel("posterior density")
    ax.set_title(f"Posterior of p from the three ABC methods\n"
                 f"(single dataset, p={p_true:.0e}, a={a}, delta={delta}, J={J})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_posterior.png", dpi=150)
    plt.close(fig)


def fig_timing():
    t3 = pd.read_csv(TABLE_DIR / "table3_timing.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = [f"{r['p']:.0e}\na={r['a']:.1f},d={r['delta']:.1f}" for _, r in t3.iterrows()]
    x = np.arange(len(labels)); w = 0.27
    ax.bar(x - w, t3["ABC-MCMC"], w, label="ABC-MCMC", color=METHOD_COLORS["ABC-MCMC"])
    ax.bar(x, t3["GPS-ABC"], w, label="GPS-ABC (GP, budget 300)", color=METHOD_COLORS["GPS-ABC"])
    ax.bar(x + w, t3["DNN-ABC"], w, label="DNN-ABC (ours)", color=METHOD_COLORS["DNN-ABC"])
    ax.set_yscale("log")
    ax.set_ylabel("seconds / 100 MCMC iterations (log scale)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("Per-iteration cost across 3-D regimes: surrogate ABC vs exact ABC-MCMC")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_timing.png", dpi=150)
    plt.close(fig)


def main():
    dnn, gp, df = _load()
    print("fig_uncertainty..."); fig_uncertainty(dnn, gp, df)
    print("fig_table1_mse..."); fig_table1()
    print("fig_timing..."); fig_timing()
    print("fig_posterior (runs a few short chains)..."); fig_posterior(dnn, gp)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
