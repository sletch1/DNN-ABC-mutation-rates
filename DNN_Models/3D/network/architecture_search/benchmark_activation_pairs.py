"""All 100 ordered activation pairs for the deployed two-layer surrogate.

`benchmark_activations.py` varies a single activation used in both hidden
layers. This script relaxes that: the 64-unit layer and the 32-unit layer may
use DIFFERENT activations, giving 10 x 10 = 100 ordered combinations. The
motivating question is whether a mixed assignment -- say a smooth unbounded
function on the wide layer and a saturating one on the narrow layer -- fits the
response surface better than any single activation used throughout.

WHY THIS NEEDS A TWO-STAGE DESIGN, AND NOT A LEADERBOARD
--------------------------------------------------------
Taking the best of 100 noisy measurements is not a measurement of the best
configuration; it is a measurement of which configuration got the luckiest
seeds. With 100 candidates and a per-fit standard deviation of the order of the
between-candidate spread, the minimum of the 100 is biased downward and will
not reproduce -- the winner's curse. Reporting that minimum as "the best
activation pair" is precisely the error this project criticises elsewhere
(see the architecture-family comparison, where 46 of 54 comparisons are ties).

So selection and evaluation are separated, and they never share a seed:

  STAGE 1 (screen).   All 100 pairs, `--screen-seeds` seeds each, seeds
                      0, 1, ... Ranks candidates. Its numbers are used ONLY to
                      choose who advances; none are reported as results.

  STAGE 2 (confirm).  The top `--finalists` pairs from stage 1, plus two
                      reference points that always advance -- the incumbent
                      (gelu, gelu) and the best homogeneous pair -- are re-fit
                      on `--confirm-seeds` FRESH seeds (100, 101, ...). Because
                      these seeds played no part in selection, stage-2 numbers
                      are an unbiased estimate of each candidate's accuracy.

The verdict compares the stage-2 spread against the across-seed standard
deviation. If the spread does not clear it, the honest conclusion is that the
100 pairs are indistinguishable on this surface, and the activation should be
chosen on grounds other than measured accuracy: smoothness of the surrogate in
its inputs (which matters if the sampler is later made gradient-based) and
robustness to dead units in a network this small.

A further diagnostic is printed regardless: the shrinkage between each
finalist's stage-1 and stage-2 MSE. Large positive shrinkage across the board
is direct evidence that stage-1 rankings were seed noise.

Usage:
    python benchmark_activation_pairs.py                       # full run
    python benchmark_activation_pairs.py --screen-seeds 1 --finalists 5
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _d in (_ROOT, _ROOT / "network", _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd

from model import _ACT
from paths import DATA, LOG_DIR
from train import load_splits, train_model, calibrate_conformal, TEST_REPS, summary_from_cultures
from surrogates import DNNSurrogate3D

BASE_ARCH = dict(kind="mlp", hidden=(64, 32))
SCREEN_SEED0, CONFIRM_SEED0 = 0, 100      # disjoint by construction


def fit_and_score(acts, tr, va, te, seed):
    """One fit at one seed. Returns (mse on held-out design means, coverage)."""
    model, xs, ys = train_model(tr[0], tr[1], va[0], va[1],
                                arch=dict(BASE_ARCH, activation=list(acts)),
                                seed=seed)
    uncal = DNNSurrogate3D(model, xs, ys, sd_scale=1.0, raw_inputs=False)
    sd_scale = calibrate_conformal(uncal, va[0], va[1])
    surr = DNNSurrogate3D(model, xs, ys, sd_scale=sd_scale, raw_inputs=False)
    X, y, design = te
    mu, sd = surr.predict(X)
    df = pd.DataFrame({"design": design, "y": y, "mu": mu, "sd": sd})
    g = df.groupby("design").agg(y=("y", "mean"), mu=("mu", "mean"))
    mse = float(np.mean((g["mu"] - g["y"]) ** 2))
    lo, hi = df["mu"] - 1.959964 * df["sd"], df["mu"] + 1.959964 * df["sd"]
    return mse, float(np.mean((df["y"] >= lo) & (df["y"] <= hi)))


def run_stage(pairs, seeds, tr, va, te, label):
    rows = []
    for i, acts in enumerate(pairs, 1):
        mses, covs, t0 = [], [], time.time()
        for s in seeds:
            m, c = fit_and_score(acts, tr, va, te, s)
            mses.append(m); covs.append(c)
        rows.append(dict(act1=acts[0], act2=acts[1],
                         mse=float(np.mean(mses)), sd=float(np.std(mses)),
                         cover95=float(np.mean(covs)),
                         secs=(time.time() - t0) / len(seeds)))
        print(f"  [{label} {i}/{len(pairs)}] {acts[0]:>9s} -> {acts[1]:<9s} "
              f"mse={rows[-1]['mse']:.3e} +/-{rows[-1]['sd']:.1e}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-seeds", type=int, default=2)
    ap.add_argument("--confirm-seeds", type=int, default=5)
    ap.add_argument("--finalists", type=int, default=8)
    args = ap.parse_args()

    tr, va, te = load_splits(DATA)
    df_raw = pd.read_csv(DATA)
    y = np.log10(summary_from_cultures(df_raw))
    var_within = df_raw.assign(y=y).groupby("design")["y"].var(ddof=1).mean()
    n_test_reps = df_raw[df_raw["rep"].isin(TEST_REPS)].groupby("design").size().mean()
    floor = var_within / n_test_reps

    names = list(_ACT)
    pairs = list(itertools.product(names, names))
    print(f"irreducible floor = {floor:.3e}")
    print(f"STAGE 1: {len(pairs)} pairs x {args.screen_seeds} seeds\n")

    screen = run_stage(pairs, list(range(SCREEN_SEED0, SCREEN_SEED0 + args.screen_seeds)),
                       tr, va, te, "screen").sort_values("mse").reset_index(drop=True)

    # Finalists: the top rows, plus two reference points that always advance so
    # the confirmation stage can be read against something meaningful.
    finalists = [tuple(r) for r in screen[["act1", "act2"]].head(args.finalists).values]
    for ref in [("gelu", "gelu")]:
        if ref not in finalists:
            finalists.append(ref)
    homo = screen[screen.act1 == screen.act2]
    if len(homo):
        best_homo = (homo.iloc[0]["act1"], homo.iloc[0]["act2"])
        if best_homo not in finalists:
            finalists.append(best_homo)

    print(f"\nSTAGE 2: {len(finalists)} finalists x {args.confirm_seeds} FRESH seeds\n")
    confirm = run_stage(finalists,
                        list(range(CONFIRM_SEED0, CONFIRM_SEED0 + args.confirm_seeds)),
                        tr, va, te, "confirm").sort_values("mse").reset_index(drop=True)

    best, worst = confirm.iloc[0], confirm.iloc[-1]
    typ_sd = float(confirm["sd"].median())
    gap = float(worst["mse"] - best["mse"])
    resolved = gap > 3 * typ_sd

    # Shrinkage: stage-1 rank optimism, measured directly.
    s1 = screen.set_index(["act1", "act2"])["mse"]
    shrink = [(float(confirm.iloc[i]["mse"]) - float(s1.loc[(confirm.iloc[i]["act1"],
                                                             confirm.iloc[i]["act2"])]))
              for i in range(len(confirm))]
    mean_shrink = float(np.mean(shrink))

    lines = ["# 3-D two-stage surrogate: mixed activation pairs\n",
             f"Architecture fixed at the deployed `mlp 64-32`; the two hidden layers may "
             f"use different activations. All {len(pairs)} ordered pairs were screened on "
             f"{args.screen_seeds} seed(s); the top {args.finalists} plus reference points "
             f"were then re-fit on {args.confirm_seeds} **fresh** seeds that played no part "
             f"in selection. Only the stage-2 numbers below are results; stage-1 was "
             f"selection only. Irreducible floor {floor:.3e}.\n",
             "## Stage 2 (fresh seeds -- the reportable numbers)\n",
             "| layer 1 (64) | layer 2 (32) | mse_mean | ±sd (seeds) | × floor | cover95 |",
             "|---|---|---|---|---|---|"]
    for _, r in confirm.iterrows():
        lines.append(f"| `{r['act1']}` | `{r['act2']}` | {r['mse']:.3e} | {r['sd']:.1e} | "
                     f"{r['mse']/floor:.3f} | {r['cover95']:.3f} |")

    lines.append(f"\n**Stage-2 spread (best to worst):** {gap:.2e} "
                 f"({100*gap/best['mse']:.1f}% of the best). "
                 f"**Typical across-seed sd:** {typ_sd:.2e}.")
    lines.append(f"\n**Selection optimism:** finalists were on average "
                 f"{mean_shrink:+.2e} worse on fresh seeds than on the seeds that "
                 f"selected them"
                 + (" -- direct evidence that the stage-1 ranking was substantially "
                    "seed noise." if mean_shrink > 0 else "."))
    if resolved:
        lines.append(f"\n**Verdict: resolved.** `{best['act1']}`->`{best['act2']}` is "
                     f"genuinely ahead of `{worst['act1']}`->`{worst['act2']}` on seeds "
                     f"not used to pick it.")
    else:
        lines.append(f"\n**Verdict: NOT resolved.** On seeds that played no part in "
                     f"selection, the finalists are within three across-seed standard "
                     f"deviations of one another. Mixing activations across layers buys "
                     f"nothing measurable on this surface, and neither does the choice of "
                     f"activation itself. The deployed activation should therefore be "
                     f"justified on structural grounds -- smoothness in the inputs, and "
                     f"no dead units in a small network -- not on a leaderboard position.")

    lines += ["\n## Stage 1 (screening only -- NOT results)\n",
              "| layer 1 | layer 2 | mse (screen) |", "|---|---|---|"]
    for _, r in screen.head(15).iterrows():
        lines.append(f"| `{r['act1']}` | `{r['act2']}` | {r['mse']:.3e} |")
    lines.append(f"\n_{len(screen)} pairs screened; showing the top 15._")

    (LOG_DIR / "benchmark_activation_pairs.md").write_text("\n".join(lines) + "\n")
    screen.to_csv(LOG_DIR / "benchmark_activation_pairs_screen.csv", index=False)
    confirm.to_csv(LOG_DIR / "benchmark_activation_pairs_confirm.csv", index=False)
    print("\n" + "\n".join(lines[-3:]))
    print(f"\nwritten -> {LOG_DIR/'benchmark_activation_pairs.md'}")


if __name__ == "__main__":
    main()
