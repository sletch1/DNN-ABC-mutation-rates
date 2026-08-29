"""Round 2: how SMALL can the 3-D surrogate be and still reach the noise floor?

Round 1 (benchmark_arch.py) found every architecture tied at ~1.08x the
irreducible floor -- a 256-128-64 network and a 128-64 network are
indistinguishable, and the residual stack the retired (p, a, delta) study used
buys nothing. Capacity is not the binding constraint on this surface.

That inverts the design question. A surrogate exists to be QUERIED, once per
MCMC iteration, tens of thousands of times per fit. Given two models that both
sit at the noise floor, the smaller one is strictly better: same accuracy, lower
latency. So this script walks capacity DOWN until accuracy visibly departs from
the floor, and reports the smallest network that still reaches it -- together
with measured query latency, which is the thing the choice actually trades.

A linear model on the four inputs is included as the bottom of the scale: it is
the "is a network needed at all?" control.

Usage:
    python benchmark_round2.py [--seeds 3]
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

from benchmark_arch import load_splits, train_one, evaluate, predict, TEST_REPS
from model import build
from paths import DATA, LOG_DIR


def query_latency(model, xs, ys, X, n=2000):
    """Microseconds per single-point prediction -- the MCMC-loop cost."""
    pt = X[:1]
    predict(model, xs, ys, pt)                      # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        predict(model, xs, ys, pt)
    return (time.perf_counter() - t0) / n * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    y = np.log10(df["d_bar"])
    floor = df.assign(y=y).groupby("design")["y"].var(ddof=1).mean() / \
        df[df["rep"].isin(TEST_REPS)].groupby("design").size().mean()
    signal = df.assign(y=y).groupby("design")["y"].mean().var(ddof=1)
    print(f"irreducible floor = {floor:.3e}   max R^2 = {1 - floor/signal:.5f}\n")

    tr, va, te = load_splits(DATA, use_derived=True)

    ladder = [
        ("256-128-64", dict(kind="mlp", hidden=(256, 128, 64), activation="gelu")),
        ("128-64",     dict(kind="mlp", hidden=(128, 64), activation="gelu")),
        ("64-32",      dict(kind="mlp", hidden=(64, 32), activation="gelu")),
        ("32-16",      dict(kind="mlp", hidden=(32, 16), activation="gelu")),
        ("16-8",       dict(kind="mlp", hidden=(16, 8), activation="gelu")),
        ("8-4",        dict(kind="mlp", hidden=(8, 4), activation="gelu")),
        ("linear",     dict(kind="mlp", hidden=(), activation="gelu")),
    ]

    rows = []
    for name, spec in ladder:
        accs, lat = [], []
        for s in range(args.seeds):
            model, xs, ys, _ = train_one(spec, tr, va, seed=s)
            accs.append(evaluate(model, xs, ys, te))
            if s == 0:
                lat.append(query_latency(model, xs, ys, te[0]))
        agg = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
        agg.update(name=name,
                   params=sum(p.numel() for p in build(in_dim=tr[0].shape[1], **spec).parameters()),
                   ratio=agg["mse_mean"] / floor, r2=1 - agg["mse_mean"] / signal,
                   us=lat[0] if lat else np.nan)
        rows.append(agg)
        print(f"{name:12s} params={agg['params']:6d}  mse_mean={agg['mse_mean']:.3e}  "
              f"={agg['ratio']:.2f}x floor  R2={agg['r2']:.5f}  cov={agg['cover95']:.3f}  "
              f"{agg['us']:.0f} us/query", flush=True)

    d = pd.DataFrame(rows)
    # "reaches the floor" = within 10% of the best observed ratio
    thresh = d.ratio.min() * 1.10
    ok = d[d.ratio <= thresh]
    pick = ok.loc[ok.params.idxmin()]

    lines = ["# 3-D two-stage surrogate: capacity floor (round 2)\n",
             f"Irreducible floor on `mse_mean` = **{floor:.3e}** (max achievable "
             f"R^2 = {1 - floor/signal:.5f}). Each row is the mean of {args.seeds} seeds, "
             "all with the derived `log10(p_eff)` feature.\n",
             "| hidden | params | mse_mean | x floor | R^2 | cover95 | us/query |",
             "|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        lines.append(f"| {r['name']} | {int(r['params'])} | {r['mse_mean']:.3e} | "
                     f"{r['ratio']:.2f} | {r['r2']:.5f} | {r['cover95']:.3f} | {r['us']:.0f} |")
    lines += [f"\n**Smallest network within 10% of the best: `{pick['name']}` "
              f"({int(pick['params'])} parameters, {pick['ratio']:.2f}x floor, "
              f"R^2 = {pick['r2']:.5f}, {pick['us']:.0f} us/query).**\n",
              "The linear row is the control: if it were competitive, no network would be "
              "justified at all. Any row whose `x floor` is near 1.0 is answering as well as "
              "the data permits, so among those the choice is purely about query cost.\n"]
    (LOG_DIR / "benchmark_round2.md").write_text("\n".join(lines) + "\n")
    d.to_csv(LOG_DIR / "benchmark_round2.csv", index=False)
    print("\n" + "\n".join(lines[-3:]))
    print(f"written -> {LOG_DIR/'benchmark_round2.md'}")


if __name__ == "__main__":
    main()
