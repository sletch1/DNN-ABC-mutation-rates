"""Architecture search for the 3-D two-stage surrogate.

WHY THIS EXISTS. The 1-D study found that the architecture choice which actually
mattered was not the activation but the presence of BatchNorm (removing it cut
error ~11x). Rather than assume that lesson transfers, this script re-runs a
controlled comparison on the two-stage data and writes the numbers to
results/logs/benchmark_arch.md, so every architectural claim in model.py and in
the manuscript is backed by a run in this repo.

WHAT IS COMPARED, each averaged over `--seeds` random seeds:
  - capacity:      hidden-layer widths and depths, plain MLP vs residual MLP
  - activation:    gelu / silu / relu / tanh
  - normalisation: none vs LayerNorm  (BatchNorm is not offered -- see model.py)
  - THE DERIVED FEATURE: with vs without log10(p_eff). This is the ablation the
    module docstring of model.py rests on.

SPLIT. By replicate, so every design point appears in every split and no
parameter combination leaks between them:
    train = reps 1-5, val = reps 6-8 (early stopping), test = reps 9-10.

METRICS.
  mse_mean : MSE of the predicted mean against the held-out DESIGN-POINT MEAN of
             log10(d_bar). This is the number that matters for the surrogate's
             job -- it measures the fitted surface, with replicate noise averaged
             out, and is directly comparable to the 1-D study's "mean-curve MSE".
  mse_obs  : MSE against individual held-out replicates (includes irreducible
             noise, so it can never approach zero).
  nll      : Gaussian NLL on held-out replicates -- scores the variance head too.
  cover95  : fraction of held-out replicates inside mean +- 1.96*sd. A model can
             have a good mean and still be useless in ABC if this is far from
             0.95, because the acceptance step consumes the predictive variance.

Usage:
    python benchmark_arch.py                 # full sweep, 3 seeds
    python benchmark_arch.py --seeds 1 --quick
"""

import argparse
import json
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

from model import build, gaussian_nll, Standardizer, add_derived
from paths import DATA, LOG_DIR

Z_975 = 1.959964
TRAIN_REPS, VAL_REPS, TEST_REPS = {1, 2, 3, 4, 5}, {6, 7, 8}, {9, 10}


def load_splits(csv_path, use_derived=True):
    """Return (X, y, design_id) per split. X is (n,3) or (n,4) with the derived feature."""
    df = pd.read_csv(csv_path)
    X = np.column_stack([np.log10(df["p1"]), np.log10(df["p2"]), df["tau"]]).astype(np.float32)
    if use_derived:
        X = add_derived(X, tp=float(df["tp"].iloc[0]))
    y = np.log10(df["d_bar"].to_numpy()).astype(np.float32)
    rep, design = df["rep"].to_numpy(), df["design"].to_numpy()
    sub = lambda r: (X[np.isin(rep, list(r))], y[np.isin(rep, list(r))], design[np.isin(rep, list(r))])
    return sub(TRAIN_REPS), sub(VAL_REPS), sub(TEST_REPS)


def _t(a, col=False):
    t = torch.tensor(np.asarray(a), dtype=torch.float32)
    return t.unsqueeze(1) if col else t


def train_one(spec, tr, va, seed=0, epochs=600, patience=40, warmup=40, bs=256):
    """Fit one architecture. Returns (model, x_scaler, y_scaler, epochs_run).

    The loss switches from MSE to Gaussian NLL after `warmup` epochs: letting the
    mean head settle first stops the variance head from explaining away a
    badly-fit mean by simply inflating sigma, which is the classic failure mode
    of training a heteroscedastic net from scratch.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    (Xtr, ytr, _), (Xva, yva, _) = tr, va
    xs = Standardizer().fit(_t(Xtr)); ys = Standardizer().fit(_t(ytr, col=True))
    xt, yt = xs.transform(_t(Xtr)), ys.transform(_t(ytr, col=True))
    xv, yv = xs.transform(_t(Xva)), ys.transform(_t(yva, col=True))

    model = build(in_dim=Xtr.shape[1], **spec)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=12)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt), batch_size=bs, shuffle=True)

    best, best_state, since, ran = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            mu, lv = model(xb)
            loss = torch.nn.functional.mse_loss(mu, yb) if ep < warmup else gaussian_nll(mu, lv, yb)
            loss.backward(); opt.step()
        model.eval(); ran = ep + 1
        with torch.no_grad():
            mu, lv = model(xv)
            vloss = gaussian_nll(mu, lv, yv).item()
        sched.step(vloss)
        if ep >= warmup and vloss < best - 1e-5:
            best, since = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif ep >= warmup:
            since += 1
            if since >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model.eval(), xs, ys, ran


@torch.no_grad()
def predict(model, xs, ys, X):
    mu, lv = model(xs.transform(_t(X)))
    mean = ys.inverse(mu).squeeze(1).numpy()
    sd = ys.inverse_std(torch.exp(0.5 * lv).squeeze(1)).numpy()
    return mean, sd


def evaluate(model, xs, ys, te):
    """Held-out metrics. `mse_mean` collapses replicates to the design-point mean."""
    X, y, design = te
    mean, sd = predict(model, xs, ys, X)
    df = pd.DataFrame({"design": design, "y": y, "mean": mean})
    g = df.groupby("design").agg({"y": "mean", "mean": "mean"})
    return dict(
        mse_mean=float(np.mean((g["mean"] - g["y"]) ** 2)),
        mse_obs=float(np.mean((mean - y) ** 2)),
        nll=float(np.mean(0.5 * (np.log(sd ** 2) + ((y - mean) / sd) ** 2))),
        cover95=float(np.mean(np.abs(y - mean) <= Z_975 * sd)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="short run, fewer variants")
    args = ap.parse_args()

    variants = [
        # name                                spec                                                   derived
        ("mlp 128-64 gelu            [1-D shape]", dict(kind="mlp", hidden=(128, 64), activation="gelu"), True),
        ("mlp 256-128-64 gelu",                    dict(kind="mlp", hidden=(256, 128, 64), activation="gelu"), True),
        ("mlp 128-128-64 silu",                    dict(kind="mlp", hidden=(128, 128, 64), activation="silu"), True),
        ("mlp 64-32 gelu             [small]",     dict(kind="mlp", hidden=(64, 32), activation="gelu"), True),
        ("mlp 256-128-64 relu",                    dict(kind="mlp", hidden=(256, 128, 64), activation="relu"), True),
        ("mlp 256-128-64 tanh",                    dict(kind="mlp", hidden=(256, 128, 64), activation="tanh"), True),
        ("mlp 256-128-64 gelu +LayerNorm",         dict(kind="mlp", hidden=(256, 128, 64), activation="gelu", use_ln=True), True),
        ("resmlp w128 x3 silu        [old 3-D]",   dict(kind="resmlp", width=128, n_blocks=3, activation="silu"), True),
        ("resmlp w128 x2 silu",                    dict(kind="resmlp", width=128, n_blocks=2, activation="silu"), True),
        # the derived-feature ablation: identical specs, raw 3 inputs only
        ("mlp 256-128-64 gelu   NO derived feat",  dict(kind="mlp", hidden=(256, 128, 64), activation="gelu"), False),
        ("resmlp w128 x3 silu   NO derived feat",  dict(kind="resmlp", width=128, n_blocks=3, activation="silu"), False),
    ]
    if args.quick:
        variants = variants[:3] + variants[-2:]

    # Irreducible floor: the test target is itself a 2-replicate mean, so it
    # carries sampling noise E[sigma^2]/n_test_reps that NO model can predict
    # away. Every mse_mean below must be read against this number -- if the best
    # architecture sits at ~1x the floor, the comparison is saturated and the
    # differences between rows are noise, not skill.
    _df = pd.read_csv(DATA); _y = np.log10(_df["d_bar"])
    _vw = _df.assign(y=_y).groupby("design")["y"].var(ddof=1).mean()
    _nte = _df[_df["rep"].isin(TEST_REPS)].groupby("design").size().mean()
    FLOOR = _vw / _nte
    SIGNAL = _df.assign(y=_y).groupby("design")["y"].mean().var(ddof=1)
    print(f"irreducible floor on mse_mean = {FLOOR:.3e}  "
          f"(max achievable R^2 = {1 - FLOOR/SIGNAL:.5f})\n")

    cache = {True: load_splits(DATA, True), False: load_splits(DATA, False)}
    print(f"train n={len(cache[True][0][1])}  val n={len(cache[True][1][1])}  "
          f"test n={len(cache[True][2][1])}\n")

    rows = []
    for name, spec, derived in variants:
        tr, va, te = cache[derived]
        accs, t0 = [], time.time()
        for s in range(args.seeds):
            model, xs, ys, ran = train_one(spec, tr, va, seed=s)
            accs.append(evaluate(model, xs, ys, te))
        agg = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
        agg["sd_mse_mean"] = float(np.std([a["mse_mean"] for a in accs]))
        agg.update(name=name, derived=derived, secs=(time.time() - t0) / args.seeds,
                   params=sum(p.numel() for p in build(in_dim=tr[0].shape[1], **spec).parameters()))
        rows.append(agg)
        print(f"{name:44s} mse_mean={agg['mse_mean']:.3e}  nll={agg['nll']:+.3f}  "
              f"cov={agg['cover95']:.3f}  {agg['secs']:.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values("mse_mean")
    best = df.iloc[0]

    lines = ["# 3-D two-stage surrogate: architecture search\n",
             f"Data: `{DATA.name}`, split by replicate (train 1-5 / val 6-8 / test 9-10). "
             f"Each row averaged over {args.seeds} seeds.\n",
             "`mse_mean` is the MSE of the predicted mean against the held-out "
             "**design-point mean** of log10(d_bar) -- the fitted surface, with replicate "
             "noise averaged out. `cover95` should sit near 0.95.\n",
             "| architecture | derived feat | params | mse_mean | ±sd | mse_obs | NLL | cover95 | s/fit |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['name']} | {'yes' if r['derived'] else 'NO'} | {int(r['params'])} | "
                     f"{r['mse_mean']:.3e} | {r['sd_mse_mean']:.1e} | {r['mse_obs']:.3e} | "
                     f"{r['nll']:+.3f} | {r['cover95']:.3f} | {r['secs']:.0f} |")
    lines.append(f"\n**Irreducible floor on `mse_mean` = {FLOOR:.3e}** "
                 f"(the test target is a {_nte:.0f}-replicate mean, so it carries "
                 f"E[sigma^2]/{_nte:.0f} of sampling noise no model can predict away). "
                 f"Max achievable R^2 = {1 - FLOOR/SIGNAL:.5f}.\n")
    lines.append(f"**Best by mse_mean: `{best['name']}`** "
                 f"(mse_mean {best['mse_mean']:.3e} = **{best['mse_mean']/FLOOR:.2f}x the floor**, "
                 f"R^2 = {1 - best['mse_mean']/SIGNAL:.5f}, coverage {best['cover95']:.3f}).\n")
    spread = df.mse_mean.max() / df.mse_mean.min()
    lines.append(f"**Read this table with care.** Best and worst differ by only "
                 f"{spread:.2f}x while the best is {best['mse_mean']/FLOOR:.2f}x the floor, so the "
                 f"architectures are effectively tied: this surface is easy relative to the "
                 f"replicate noise, and capacity is not the binding constraint. The design "
                 f"question is therefore how SMALL a model still reaches the floor "
                 f"(see benchmark_round2.py), because the surrogate's value is query speed "
                 f"inside the MCMC loop.\n")

    # the ablation, stated explicitly
    for base in ("mlp 256-128-64 gelu", "resmlp w128 x3 silu        [old 3-D]"):
        w = df[df.name == base]
        wo = df[df.name.str.startswith(base.split()[0] + " " + base.split()[1]) & (~df.derived)]
        if len(w) and len(wo):
            lines.append(f"- Derived feature on `{base.split('[')[0].strip()}`: "
                         f"mse_mean {float(wo.mse_mean.iloc[0]):.3e} (without) -> "
                         f"{float(w.mse_mean.iloc[0]):.3e} (with), "
                         f"a {float(wo.mse_mean.iloc[0]) / float(w.mse_mean.iloc[0]):.2f}x change.")

    (LOG_DIR / "benchmark_arch.md").write_text("\n".join(lines) + "\n")
    df.to_csv(LOG_DIR / "benchmark_arch.csv", index=False)
    print("\n" + "\n".join(lines[-4:]))
    print(f"\nwritten -> {LOG_DIR/'benchmark_arch.md'}")


if __name__ == "__main__":
    main()
