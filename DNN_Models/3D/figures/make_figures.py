"""Result figures for the 3-D two-stage study.

Each figure answers one question a reader will ask, and none of them are
decorative:

  fig_data_structure.png  What does the ground truth actually look like, and why
                          is this a hard inference problem? Shows how strongly
                          d_bar responds to each parameter -- the asymmetry
                          (p2 strong, p1 weak, tau weakest) that drives every
                          downstream result.
  fig_noise.png           Is the replicate noise heteroscedastic? This is the
                          justification for the two-headed model: a single
                          homoscedastic term, all a GP offers, cannot represent
                          a 292x spread in noise that tracks the target.
  fig_capacity.png        The architecture search against the irreducible noise
                          floor -- why the network is small and why a linear
                          model is nonetheless not enough.
  fig_calibration.png     Does the predictive uncertainty mean what it claims?
                          Nominal vs empirical coverage after conformal scaling.

Figures that depend on ABC results (posteriors, timing) are produced only when
abc/run_experiments.py has been run and its tables exist; they are skipped with
a message otherwise, so this script is safe to run at any stage.

Usage:
    python make_figures.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import DATA, FIG_DIR, LOG_DIR, TABLE_DIR, MODEL_DIR

Z_975 = 1.959964
TEST_REPS = {9, 10}


def fig_data_structure(df):
    """d_bar against each parameter, with the correlation that makes the point."""
    y = np.log10(df.d_bar)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (col, lab, x) in zip(axes, [("p2", "log10(p2)", np.log10(df.p2)),
                                        ("p1", "log10(p1)", np.log10(df.p1)),
                                        ("tau", "tau", df.tau)]):
        ax.scatter(x, y, s=3, alpha=0.08, color="tab:blue", rasterized=True)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(lab)
        ax.set_title(f"corr = {r:+.3f}", fontsize=10)
    axes[0].set_ylabel("log10(d_bar)")
    fig.suptitle("How much does the summary statistic know about each parameter?", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_data_structure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_noise(df):
    """Within-design replicate sd against the design's mean -- the heteroscedasticity."""
    g = df.assign(y=np.log10(df.d_bar)).groupby("design")["y"]
    m, s = g.mean(), g.std()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(m, s, s=8, alpha=0.35, color="tab:red")
    ax.set_xlabel("design-point mean of log10(d_bar)")
    ax.set_ylabel("within-design replicate sd")
    ax.set_yscale("log")
    ax.set_title(f"Replicate noise spans {s.max()/s.min():.0f}x and tracks the target "
                 f"(corr = {np.corrcoef(m, s)[0,1]:+.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_noise.png", dpi=150)
    plt.close(fig)


def fig_capacity():
    """Accuracy vs capacity against the irreducible floor."""
    f = LOG_DIR / "benchmark_round2.csv"
    if not f.exists():
        print("skip fig_capacity: run network/architecture_search/benchmark_round2.py first")
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    net = d[d.name != "linear"]
    ax.plot(net.params, net.ratio, "o-", color="tab:blue", label="MLP")
    lin = d[d.name == "linear"]
    if len(lin):
        ax.plot(lin.params, lin.ratio, "s", color="tab:red", ms=9, label="linear control")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="irreducible noise floor")
    for _, r in d.iterrows():
        ax.annotate(r["name"], (r.params, r.ratio), fontsize=7,
                    textcoords="offset points", xytext=(4, 5))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("parameters"); ax.set_ylabel("mse_mean / irreducible floor")
    ax.set_title("Every network above ~700 parameters is at the noise floor", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_capacity.png", dpi=150)
    plt.close(fig)


def fig_calibration(df):
    """Nominal vs empirical coverage of the calibrated predictive interval."""
    ckpt = MODEL_DIR / "surrogate_3d.pt"
    if not ckpt.exists():
        print("skip fig_calibration: run network/train.py first")
        return
    from scipy.stats import norm
    from train import load_surrogate, load_splits
    surr = load_surrogate(ckpt)
    (_, _, _), (_, _, _), (Xte, yte, _) = load_splits(DATA)
    # load_splits already appended the derived column; predict on it directly
    surr.raw_inputs = False
    mean, sd = surr.predict(Xte)
    levels = np.linspace(0.05, 0.99, 40)
    emp = [np.mean(np.abs(yte - mean) <= norm.ppf(0.5 + l / 2) * sd) for l in levels]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(levels, emp, color="tab:blue", lw=2, label="DNN + conformal")
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Predictive-interval calibration (held-out)"); ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_calibration.png", dpi=150)
    plt.close(fig)


def fig_posterior():
    """Posterior marginals per method, if the ABC experiments have been run."""
    f = LOG_DIR / "raw_replicates.csv"
    if not f.exists():
        print("skip fig_posterior: run abc/run_experiments.py first")
        return
    d = pd.read_csv(f)
    methods = [m for m in ("ABC-MCMC", "GPS-ABC", "DNN-ABC") if f"{m}_p2" in d.columns]
    params = ["p1", "p2", "tau"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, p in zip(axes, params):
        for m in methods:
            est = d[f"{m}_{p}"].to_numpy(float)
            ax.hist(est[np.isfinite(est)], bins=25, alpha=0.45, label=m)
        tv = d[f"{p}_true"].iloc[0]
        ax.axvline(tv, color="k", ls="--", lw=1)
        ax.set_xlabel(p)
        if p in ("p1", "p2"):
            ax.set_xscale("log")
    axes[0].set_ylabel("count"); axes[0].legend(fontsize=8)
    fig.suptitle("Posterior-mean estimates across replicates (dashed = truth)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_posterior.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(DATA)
    fig_data_structure(df)
    fig_noise(df)
    fig_capacity()
    fig_calibration(df)
    fig_posterior()
    print(f"figures written to {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
