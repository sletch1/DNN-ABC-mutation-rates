"""Controlled comparison of activation functions for the two-stage surrogate.

WHAT THIS ANSWERS. Every other choice is held fixed -- the deployed 64-32 funnel
MLP, the same Gaussian-NLL objective, the same split-by-replicate data, the same
optimizer, schedule and early-stopping rule -- and only the activation varies.
Ten candidates are compared, spanning the three families that behave differently
on a smooth regression surface:

    piecewise-linear   relu, leakyrelu, prelu
    smooth saturating  tanh, elu, selu
    smooth unbounded   softplus, gelu, silu, mish

WHY THESE TEN, for this surface specifically. The target E[log10 d_bar | theta]
is a smooth, non-linear function of three standardized parameters (a linear fit
reaches R^2 = 0.37, quadratic-with-interactions 0.965), fitted by a small
network in which each unit is a large fraction of total capacity. Three
properties therefore matter, and the ten span them:
  - Smoothness. A piecewise-linear activation approximates a smooth surface with
    kinks, and makes the surrogate non-differentiable in its inputs -- which
    matters if the sampler is ever made gradient-based (HMC/NUTS).
  - A live negative region. Inputs are standardized, so about half of all
    pre-activations are negative; relu discards that half outright and can
    strand units permanently (the dead-unit failure mode), which is costlier
    here than in a wide network.
  - Stable gradients for the variance head. Gaussian NLL couples the mean and
    log-variance heads, so an activation that destabilises training shows up as
    a miscalibrated sigma, not just a worse mean.

HOW TO READ THE OUTPUT -- this is the important part. Every row is reported as a
mean over `--seeds` random seeds together with the standard deviation ACROSS
those seeds, and scored against the irreducible noise floor (the held-out target
is a 2-replicate mean, so it carries E[sigma^2]/2 of noise that no model can
predict away). A difference between two rows is only real if it is large
relative to that seed spread. On this surface the expected outcome is that most
candidates tie: the script prints an explicit verdict saying whether the ranking
is resolved or is seed noise, rather than leaving the reader to assume the top
row is meaningfully best.

Usage:
    python benchmark_activations.py --seeds 5
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _d in (_ROOT, _ROOT / "network", _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd
import torch

from model import build, _ACT
from paths import DATA, LOG_DIR
from train import load_splits, train_model, calibrate_conformal, TEST_REPS, summary_from_cultures
from surrogates import DNNSurrogate3D

# The deployed architecture; only `activation` is varied below.
BASE_ARCH = dict(kind="mlp", hidden=(64, 32))

FAMILY = {
    "relu": "piecewise-linear", "leakyrelu": "piecewise-linear", "prelu": "piecewise-linear",
    "tanh": "smooth saturating", "elu": "smooth saturating", "selu": "smooth saturating",
    "softplus": "smooth unbounded", "gelu": "smooth unbounded",
    "silu": "smooth unbounded", "mish": "smooth unbounded",
}


def evaluate(model, xs, ys, sd_scale, split):
    """Score one fitted model on held-out DESIGN-POINT MEANS.

    Averaging replicates within a design point removes the replicate noise that
    no model could predict, so this isolates the fitted surface itself. Returns
    the MSE of that surface, plus coverage of the calibrated 95% interval.
    """
    X, y, design = split
    surr = DNNSurrogate3D(model, xs, ys, sd_scale=sd_scale, raw_inputs=False)
    mu, sd = surr.predict(X)
    df = pd.DataFrame({"design": design, "y": y, "mu": mu, "sd": sd})
    g = df.groupby("design").agg(y=("y", "mean"), mu=("mu", "mean"), sd=("sd", "mean"))
    mse_mean = float(np.mean((g["mu"] - g["y"]) ** 2))
    lo, hi = df["mu"] - 1.959964 * df["sd"], df["mu"] + 1.959964 * df["sd"]
    cover = float(np.mean((df["y"] >= lo) & (df["y"] <= hi)))
    return mse_mean, cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    tr, va, te = load_splits(DATA)

    # Irreducible floor: the test target is a 2-replicate mean.
    df = pd.read_csv(DATA)
    y = np.log10(summary_from_cultures(df))
    var_within = df.assign(y=y).groupby("design")["y"].var(ddof=1).mean()
    n_test_reps = df[df["rep"].isin(TEST_REPS)].groupby("design").size().mean()
    floor = var_within / n_test_reps
    print(f"irreducible floor = {floor:.3e}   ({args.seeds} seeds per activation)\n")

    rows = []
    for act in _ACT:
        mses, covs, t0 = [], [], time.time()
        for s in range(args.seeds):
            model, xs, ys = train_model(tr[0], tr[1], va[0], va[1],
                                        arch=dict(BASE_ARCH, activation=act), seed=s)
            uncal = DNNSurrogate3D(model, xs, ys, sd_scale=1.0, raw_inputs=False)
            sd_scale = calibrate_conformal(uncal, va[0], va[1])
            m, c = evaluate(model, xs, ys, sd_scale, te)
            mses.append(m); covs.append(c)
        row = dict(activation=act, family=FAMILY[act],
                   mse_mean=float(np.mean(mses)),
                   # ddof=1: these seeds are a sample, and the standard error of
                   # the mean (sd/sqrt(n)) is what pairwise comparisons need.
                   sd_seeds=float(np.std(mses, ddof=1)),
                   se_mean=float(np.std(mses, ddof=1) / np.sqrt(len(mses))),
                   x_floor=float(np.mean(mses)) / floor,
                   cover95=float(np.mean(covs)),
                   secs=(time.time() - t0) / args.seeds)
        # keep every per-seed value so pairwise tests can be redone from the CSV
        for i, m in enumerate(mses):
            row[f"mse_seed{i}"] = m
        rows.append(row)
        r = rows[-1]
        print(f"  {act:10s} {r['family']:18s} mse={r['mse_mean']:.3e} "
              f"+/-{r['sd_seeds']:.1e}  ={r['x_floor']:.3f}x floor  cov={r['cover95']:.3f}",
              flush=True)

    res = pd.DataFrame(rows).sort_values("mse_mean").reset_index(drop=True)
    best, worst = res.iloc[0], res.iloc[-1]

    # Is the ranking real? Compare the best-vs-worst gap to the typical
    # seed-to-seed spread. If the gap is not several times the spread, the
    # ordering is noise and any "winner" would not reproduce on new seeds.
    typ_sd = float(res["sd_seeds"].median())
    gap = float(worst["mse_mean"] - best["mse_mean"])
    resolved = gap > 3 * typ_sd

    lines = ["# 3-D two-stage surrogate: activation-function comparison\n",
             f"Architecture held fixed at the deployed `mlp 64-32`; only the activation "
             f"varies. Each row is the mean of {args.seeds} seeds, with the standard "
             f"deviation across those seeds. MSE is against held-out design-point means "
             f"of log10(d_bar); the irreducible floor is {floor:.3e}.\n",
             "| activation | family | mse_mean | ±sd (seeds) | × floor | cover95 | s/fit |",
             "|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append(f"| `{r['activation']}` | {r['family']} | {r['mse_mean']:.3e} | "
                     f"{r['sd_seeds']:.1e} | {r['x_floor']:.3f} | {r['cover95']:.3f} | "
                     f"{r['secs']:.0f} |")

    # Pairwise comparisons against the best row, in units of the standard error
    # of the difference. This is what says which rows are actually separated,
    # rather than only comparing the two extremes.
    lines.append("\n**Pairwise against the best row** (t = gap / SE of the "
                 "difference; |t| < 2 is a tie at this seed count):\n")
    lines.append("| activation | × floor | gap vs best | t |")
    lines.append("|---|---|---|---|")
    for _, r in res.iterrows():
        if r["activation"] == best["activation"]:
            lines.append(f"| `{r['activation']}` | {r['x_floor']:.3f} | — | reference |")
            continue
        diff = r["mse_mean"] - best["mse_mean"]
        se = float(np.sqrt(r["se_mean"] ** 2 + best["se_mean"] ** 2))
        lines.append(f"| `{r['activation']}` | {r['x_floor']:.3f} | {diff:.2e} | "
                     f"{diff/se:.2f} |")

    lines.append(f"\n**Best-to-worst spread:** {gap:.2e} "
                 f"({100*gap/best['mse_mean']:.1f}% of the best row's MSE). "
                 f"**Typical across-seed sd:** {typ_sd:.2e}.")
    if resolved:
        lines.append(f"\n**Verdict: the ranking is resolved.** The spread exceeds three "
                     f"times the typical seed noise, so `{best['activation']}` is "
                     f"genuinely better than `{worst['activation']}` here.")
    else:
        lines.append(f"\n**Verdict: the ranking is NOT resolved.** The best-to-worst "
                     f"spread is smaller than three times the typical across-seed "
                     f"standard deviation, so the ordering above is seed noise: on this "
                     f"surface the activation function does not measurably matter. "
                     f"Choosing between these should therefore rest on properties other "
                     f"than measured accuracy -- smoothness of the surrogate in its "
                     f"inputs, and robustness to dead units in a small network.")

    out = LOG_DIR / "benchmark_activations.md"
    out.write_text("\n".join(lines) + "\n")
    res.to_csv(LOG_DIR / "benchmark_activations.csv", index=False)
    print("\n" + "\n".join(lines[-2:]))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
