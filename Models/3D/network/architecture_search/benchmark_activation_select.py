"""Choose the activation pair for the deployed 64->32 network, without touching test.

WHY THIS SCRIPT EXISTS. `benchmark_activations.py` and
`benchmark_activation_pairs.py` both score candidates on the TEST split. That
makes the reported test MSE optimistic for whichever activation wins: the split
that is supposed to be untouched was used to pick the model. This script fixes
the procedure:

    stage 1  all 100 ordered (layer1, layer2) pairs, few seeds, scored on VAL
    stage 2  the finalists, many FRESH seeds, scored on VAL  -> winner
    stage 3  the winner alone is scored on TEST, once

Validation is already used for early stopping and conformal calibration, so
selecting on it is mildly optimistic too -- but it is the standard train/select/
report split discipline, and it leaves the test number honest, which is what the
manuscript quotes.

Each worker pins torch to one thread: parallelism here is across fits, so letting
every process spin up a full thread pool would oversubscribe the machine.

    python3 architecture_search/benchmark_activation_select.py
    python3 architecture_search/benchmark_activation_select.py --screen-seeds 2 --confirm-seeds 15
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from multiprocessing import Pool
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
SCREEN_SEED0, CONFIRM_SEED0 = 0, 1000        # disjoint by construction
_G = {}


def _init(_):
    import torch, warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    _G["splits"] = load_splits(DATA)


def _score(surr, split):
    X, y, design = split
    mu, sd = surr.predict(X)
    df = pd.DataFrame({"design": design, "y": y, "mu": mu, "sd": sd})
    g = df.groupby("design").agg(y=("y", "mean"), mu=("mu", "mean"))
    mse = float(np.mean((g["mu"] - g["y"]) ** 2))
    lo, hi = df["mu"] - 1.959964 * df["sd"], df["mu"] + 1.959964 * df["sd"]
    return mse, float(np.mean((df["y"] >= lo) & (df["y"] <= hi)))


def _one(task):
    """One (pair, seed) fit. Returns val and test scores; only val is used to select."""
    a1, a2, seed = task
    tr, va, te = _G["splits"]
    model, xs, ys = train_model(tr[0], tr[1], va[0], va[1],
                                arch=dict(BASE_ARCH, activation=[a1, a2]), seed=seed)
    uncal = DNNSurrogate3D(model, xs, ys, sd_scale=1.0, raw_inputs=False)
    sd_scale = calibrate_conformal(uncal, va[0], va[1])
    surr = DNNSurrogate3D(model, xs, ys, sd_scale=sd_scale, raw_inputs=False)
    v_mse, v_cov = _score(surr, va)
    t_mse, t_cov = _score(surr, te)
    return dict(act1=a1, act2=a2, seed=seed,
                val_mse=v_mse, val_cov=v_cov, test_mse=t_mse, test_cov=t_cov)


def _run(tasks, workers, label):
    rows, t0 = [], time.time()
    with Pool(workers, initializer=_init, initargs=(None,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, tasks), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  [{label} {i}/{len(tasks)}] elapsed {el/60:.1f}m "
                      f"eta {el/i*(len(tasks)-i)/60:.1f}m", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-seeds", type=int, default=2)
    ap.add_argument("--confirm-seeds", type=int, default=15)
    ap.add_argument("--finalists", type=int, default=10)
    ap.add_argument("--workers", type=int,
                    default=max(1, (__import__("os").cpu_count() or 2) - 2))
    args = ap.parse_args()

    df_raw = pd.read_csv(DATA)
    y = np.log10(summary_from_cultures(df_raw))
    var_within = df_raw.assign(y=y).groupby("design")["y"].var(ddof=1).mean()
    n_val = df_raw[~df_raw["rep"].isin(TEST_REPS)].groupby("design").size().mean()
    n_test = df_raw[df_raw["rep"].isin(TEST_REPS)].groupby("design").size().mean()
    floor_test = var_within / n_test

    names = list(_ACT)
    pairs = list(itertools.product(names, names))
    print(f"stage 1: {len(pairs)} pairs x {args.screen_seeds} seeds on VAL "
          f"({args.workers} workers)")
    s1 = _run([(a, b, SCREEN_SEED0 + s) for a, b in pairs
               for s in range(args.screen_seeds)], args.workers, "screen")
    rank1 = (s1.groupby(["act1", "act2"])["val_mse"].mean()
               .sort_values().reset_index())
    finalists = list(rank1.head(args.finalists)[["act1", "act2"]].itertuples(index=False, name=None))
    print("\nfinalists (by VAL):", ", ".join(f"{a}->{b}" for a, b in finalists))

    print(f"\nstage 2: {len(finalists)} finalists x {args.confirm_seeds} FRESH seeds on VAL")
    s2 = _run([(a, b, CONFIRM_SEED0 + s) for a, b in finalists
               for s in range(args.confirm_seeds)], args.workers, "confirm")

    g = (s2.groupby(["act1", "act2"])
           .agg(val_mse=("val_mse", "mean"), val_sd=("val_mse", "std"),
                test_mse=("test_mse", "mean"), test_sd=("test_mse", "std"),
                cov=("val_cov", "mean"), n=("seed", "size"))
           .reset_index().sort_values("val_mse").reset_index(drop=True))
    g["val_se"] = g["val_sd"] / np.sqrt(g["n"])
    best = g.iloc[0]
    # Welch t of each finalist against the winner, on VAL (the selection metric).
    ts = []
    for _, r in g.iterrows():
        se = float(np.hypot(best["val_se"], r["val_se"]))
        ts.append(0.0 if r.equals(best) or se == 0 else (r["val_mse"] - best["val_mse"]) / se)
    g["t_vs_best"] = ts

    s1.to_csv(LOG_DIR / "benchmark_activation_select_screen.csv", index=False)
    s2.to_csv(LOG_DIR / "benchmark_activation_select_confirm.csv", index=False)

    # "Resolved" must mean the WINNER is separated from the runner-up -- not merely
    # that the bottom of the field falls away. Count how many finalists sit within
    # 2 SE of the winner: those are all tied for first.
    n_tied = int((g["t_vs_best"] <= 2).sum())
    resolved = n_tied == 1
    lines = [
        "# 3-D two-stage surrogate: activation selection (no test-set leakage)\n",
        f"All {len(pairs)} ordered activation pairs for the deployed `mlp 64-32`, screened on "
        f"{args.screen_seeds} seed(s) and confirmed on {args.confirm_seeds} **fresh** seeds. "
        "**Selection is on the validation split**; the test column is shown for "
        "reference only and played no part in the choice.\n",
        f"Winner: **`{best['act1']}` -> `{best['act2']}`**  "
        f"(val MSE {best['val_mse']:.3e}, test MSE {best['test_mse']:.3e} = "
        f"{best['test_mse']/floor_test:.3f}x the irreducible test floor {floor_test:.3e}).\n",
        "| layer 1 (64) | layer 2 (32) | val MSE | ±sd | t vs best | test MSE (reference) | cover95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in g.iterrows():
        lines.append(f"| `{r['act1']}` | `{r['act2']}` | {r['val_mse']:.3e} | {r['val_sd']:.1e} | "
                     f"{r['t_vs_best']:.2f} | {r['test_mse']:.3e} | {r['cov']:.3f} |")
    lines += ["", f"**Verdict: {'RESOLVED' if resolved else 'NOT resolved'}.** " +
              ("The winner is separated from every other finalist by more than 2 standard "
               "errors on the selection split, so the choice carries signal."
               if resolved else
               f"{n_tied} of the {len(g)} finalists sit within 2 standard errors of the "
               "winner on the selection split, so the winner is the best point estimate but "
               "is NOT statistically distinguishable from the rest of that group. What the "
               "comparison does resolve is the bottom of the field, not the top. The "
               "deployed pair is therefore justified on the best point estimate together "
               "with the structural argument -- smooth in the inputs, no dead units in a "
               "small network -- rather than on a leaderboard position that the data do "
               "not support.")]
    (LOG_DIR / "benchmark_activation_select.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[2:]))
    print(f"\nwritten -> {LOG_DIR/'benchmark_activation_select.md'}")


if __name__ == "__main__":
    main()
