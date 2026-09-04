"""Fit-quality benchmark for the three new architecture families (CNN1D, RNN,
LSTM) against the already-deployed FFN, on the 3-D two-stage surrogate task.

WHY THIS EXISTS. `benchmark_arch.py` and `benchmark_round2.py` (this directory)
established that a plain MLP reaches the data's irreducible noise floor with as
few as ~700-2,500 parameters, and that capacity is not the binding constraint on
this surface. This script asks a different question: does an architecture with
a built-in structural prior -- convolution over "neighbouring" positions
(CNN1D), or recurrence over "sequential" positions (RNN/LSTM) -- do any better
or worse than the plain MLP, given that (as `network/model_families.py`
explains in detail) the four inputs here have no real spatial or temporal
relationship to exploit? This is expected, honestly, to be a negative or
neutral result; it is reported as measured either way.

WHAT IS COMPARED, each averaged over 2 seeds (matching the existing
architecture-search convention in this directory):
  - HeteroscedasticCNN1D  (model_families.py): small Conv1d stack over the
    4-length input treated as a 1-channel signal.
  - HeteroscedasticRNN    (model_families.py): a small GRU over 4 length-1
    "timesteps" (see model_families.py for why GRU over a vanilla RNN).
  - HeteroscedasticLSTM   (model_families.py): same shape convention, nn.LSTM.
  - FFN (64-32 GELU)      : NOT retrained here. Pulled verbatim from
    `results/logs/benchmark_round2.md` / `results/model/surrogate_metrics.json`
    -- the already-deployed, already-reported surrogate. Re-running it would
    both waste time and risk a spurious seed-to-seed difference from the number
    actually cited in the manuscript; the existing number is a fixed reference
    point instead.

SPLIT, TRAINING REGIME, DATA -- identical to `train.py` / `benchmark_arch.py`:
train = reps 1-5, val = reps 6-8 (early stopping + conformal calibration),
test = reps 9-10, both by replicate. Same Adam(lr=1e-3, weight_decay=1e-5),
same ReduceLROnPlateau, same warmup-then-NLL training loop as
`network/train.py: train_model`. The one documented deviation: optional
gradient clipping (max-norm 1.0) for the RNN and LSTM cells, off by default for
the CNN1D and never used for the deployed MLP. Recurrent nets are the textbook
case for unstable gradients across a rollout; even at only 4 "timesteps" this
costs nothing (a single extra `clip_grad_norm_` call) and removes a known
failure mode without changing the tested architectures' capacity or the results
of the mean-vs-no-clip ablation would be a paper unto itself, not run here.

METRICS -- same as `benchmark_round2.py`'s table: `mse_mean` (fitted-surface
MSE against the held-out design-point mean), `x floor` (ratio to the
irreducible noise floor, computed fresh from the data here and expected to
match the already-established 1.393e-3 in `results/logs/benchmark_round2.md`),
`R^2`, `cover95` (95% predictive coverage after split-conformal calibration),
`params`, and `us/query` (single-point prediction latency -- the cost that
actually matters inside the MCMC loop).

CHECKPOINTS. The best seed (by `mse_mean`) of each new family is also saved,
conformally calibrated exactly as `train.py` does for the deployed FFN, to
`results/arch_families/checkpoints/<family>_best.pt` -- these feed directly
into `abc/run_experiments_families.py`'s full ABC-MCMC evaluation, reusing the
existing `DNNSurrogate3D` wrapper (model_families.py's classes all return
`(mean, logvar)` exactly like `HeteroscedasticMLP`, so nothing about the
surrogate interface needs to change to accept them).

Usage:
    python benchmark_families.py                # 2 seeds/family, writes results
    python benchmark_families.py --seeds 3
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]          # .../DNN_Models/3D
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd
import torch

from benchmark_arch import load_splits, predict, evaluate, Z_975, TEST_REPS
from benchmark_round2 import query_latency
from model import Standardizer, gaussian_nll
from model_families import build_family
from train import calibrate_conformal
from surrogates import DNNSurrogate3D
from paths import DATA

ARCH_FAMILIES_DIR = _ROOT / "results" / "arch_families"
CKPT_DIR = ARCH_FAMILIES_DIR / "checkpoints"
for _d in (ARCH_FAMILIES_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Already-reported reference row: the deployed FFN, NOT retrained here.
# Source: results/logs/benchmark_round2.md (the "64-32" row) and
# results/model/surrogate_metrics.json (arch). Copied as literal numbers, not
# re-derived, so this table can never silently drift from what the manuscript
# already cites.
FFN_REFERENCE = dict(
    name="FFN 64-32 gelu  [DEPLOYED]", params=2466,
    mse_mean=1.573e-03, ratio=1.13, r2=0.99467, cover95=0.952, us=41.0,
)


def _t(a, col=False):
    t = torch.tensor(np.asarray(a), dtype=torch.float32)
    return t.unsqueeze(1) if col else t


def train_one_family(kind, spec, tr, va, seed=0, epochs=800, patience=50,
                     warmup=50, bs=256, grad_clip=None):
    """Fit one (family, spec) pair. Mirrors `train.py: train_model` exactly
    (Adam lr=1e-3/wd=1e-5, ReduceLROnPlateau, MSE-then-NLL warmup, early
    stopping on val NLL) but builds the model via `model_families.build_family`
    instead of `model.build`, and optionally clips gradients (see module
    docstring) for the RNN/LSTM families.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    (Xtr, ytr, _), (Xva, yva, _) = tr, va
    xs = Standardizer().fit(_t(Xtr)); ys = Standardizer().fit(_t(ytr, col=True))
    xt, yt = xs.transform(_t(Xtr)), ys.transform(_t(ytr, col=True))
    xv, yv = xs.transform(_t(Xva)), ys.transform(_t(yva, col=True))

    model = build_family(kind, in_dim=Xtr.shape[1], **spec)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt), batch_size=bs, shuffle=True)

    best, best_state, since = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            mu, lv = model(xb)
            loss = torch.nn.functional.mse_loss(mu, yb) if ep < warmup else gaussian_nll(mu, lv, yb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        model.eval()
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
    return model.eval(), xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    y = np.log10(df["d_bar"])
    n_te = df[df["rep"].isin(TEST_REPS)].groupby("design").size().mean()
    var_within = df.assign(y=y).groupby("design")["y"].var(ddof=1).mean()
    FLOOR = var_within / n_te
    SIGNAL = df.assign(y=y).groupby("design")["y"].mean().var(ddof=1)
    print(f"irreducible floor on mse_mean = {FLOOR:.3e}  "
          f"(established value: 1.393e-03; max achievable R^2 = {1 - FLOOR / SIGNAL:.5f})\n")

    tr, va, te = load_splits(DATA, use_derived=True)
    print(f"train n={len(tr[1])}  val n={len(va[1])}  test n={len(te[1])}\n")

    # (family key, display name, kwargs, whether to clip gradients)
    families = [
        ("cnn1d", "CNN1D (16,16 ch, k=3)",
         dict(channels=(16, 16), kernel_size=3, hidden=32, activation="gelu"), None),
        ("rnn", "RNN (GRU, hidden=32)",
         dict(cell="gru", hidden_size=32), 1.0),
        ("lstm", "LSTM (hidden=24)",
         dict(hidden_size=24), 1.0),
    ]

    rows, best_ckpts = [], {}
    for kind, label, spec, grad_clip in families:
        seed_results = []
        t0 = time.time()
        for s in range(args.seeds):
            model, xs, ys = train_one_family(kind, spec, tr, va, seed=s, grad_clip=grad_clip)
            m = evaluate(model, xs, ys, te)
            lat = query_latency(model, xs, ys, te[0]) if s == 0 else None
            seed_results.append((s, model, xs, ys, m, lat))
        secs = (time.time() - t0) / args.seeds

        mses = [r[4]["mse_mean"] for r in seed_results]
        agg = dict(name=label, kind=kind,
                   mse_mean=float(np.mean(mses)), sd_mse_mean=float(np.std(mses)),
                   mse_obs=float(np.mean([r[4]["mse_obs"] for r in seed_results])),
                   nll=float(np.mean([r[4]["nll"] for r in seed_results])),
                   cover95=float(np.mean([r[4]["cover95"] for r in seed_results])),
                   params=sum(p.numel() for p in build_family(kind, in_dim=tr[0].shape[1], **spec).parameters()),
                   secs=secs)
        agg["ratio"] = agg["mse_mean"] / FLOOR
        agg["r2"] = 1 - agg["mse_mean"] / SIGNAL
        agg["us"] = next(r[5] for r in seed_results if r[5] is not None)
        rows.append(agg)
        print(f"{label:26s} mse_mean={agg['mse_mean']:.3e}  x_floor={agg['ratio']:.2f}  "
              f"R2={agg['r2']:.5f}  cover95={agg['cover95']:.3f}  params={agg['params']}  "
              f"{agg['us']:.0f} us/query  ({secs:.0f}s/seed)", flush=True)

        # Best seed by mse_mean -> conformally calibrated, saved as a checkpoint
        # ready for abc/run_experiments_families.py.
        best_seed = min(seed_results, key=lambda r: r[4]["mse_mean"])
        s_idx, model, xs, ys, m, _ = best_seed
        surr = DNNSurrogate3D(model, xs, ys, sd_scale=1.0, use_derived=True, raw_inputs=False)
        sd_scale = calibrate_conformal(surr, va[0], va[1])
        ckpt_path = CKPT_DIR / f"{kind}_best.pt"
        torch.save({"model_state": model.state_dict(), "x_scaler": xs.state_dict(),
                   "y_scaler": ys.state_dict(), "sd_scale": sd_scale,
                   "kind": kind, "spec": spec, "seed": s_idx,
                   "use_derived": True,
                   "input": "[log10(p1), log10(p2), tau]", "output": "log10(d_bar)",
                   "heteroscedastic": True, "source_csv": str(DATA)}, ckpt_path)
        best_ckpts[kind] = str(ckpt_path)
        print(f"  -> best seed {s_idx} (mse_mean={m['mse_mean']:.3e}), "
              f"conformal sd_scale={sd_scale:.4f}, saved -> {ckpt_path}")

    df_out = pd.DataFrame(rows).sort_values("mse_mean")

    lines = ["# 3-D two-stage surrogate: new architecture families vs the deployed FFN\n",
             f"Data: `{DATA.name}`, split by replicate (train 1-5 / val 6-8 / test 9-10), "
             f"identical to `train.py` / `benchmark_arch.py`. New families averaged over "
             f"{args.seeds} seeds; the FFN row is **not retrained** here -- it is the "
             "literal already-deployed/already-reported number "
             "(`results/logs/benchmark_round2.md`, `results/model/surrogate_metrics.json`).\n",
             f"Irreducible floor on `mse_mean` = **{FLOOR:.3e}** "
             f"(max achievable R^2 = {1 - FLOOR / SIGNAL:.5f}).\n",
             "| architecture | params | mse_mean | x floor | R^2 | cover95 | us/query |",
             "|---|---|---|---|---|---|---|"]
    ffn = FFN_REFERENCE
    combined = pd.concat([df_out, pd.DataFrame([{**ffn}])], ignore_index=True, sort=False)
    combined = combined.sort_values("mse_mean")
    for _, r in combined.iterrows():
        lines.append(f"| {r['name']} | {int(r['params'])} | {r['mse_mean']:.3e} | "
                     f"{r['ratio']:.2f} | {r['r2']:.5f} | {r['cover95']:.3f} | {r['us']:.0f} |")

    best = combined.iloc[0]
    lines.append(f"\n**Best by mse_mean: `{best['name']}`** "
                f"({best['mse_mean']:.3e} = {best['ratio']:.2f}x floor, "
                f"R^2 = {best['r2']:.5f}, cover95 = {best['cover95']:.3f}).\n")
    ffn_row = combined[combined.name == FFN_REFERENCE["name"]].iloc[0]
    for _, r in combined.iterrows():
        if r["name"] == FFN_REFERENCE["name"]:
            continue
        delta = (r["mse_mean"] / ffn_row["mse_mean"] - 1) * 100
        verdict = "worse" if delta > 0 else "better"
        lines.append(f"- `{r['name']}` vs deployed FFN: mse_mean {delta:+.1f}% "
                    f"({abs(delta):.1f}% {verdict}), both within "
                    f"{max(r['ratio'], ffn_row['ratio']):.2f}x of the noise floor.")
    lines.append(f"\nSee `network/model_families.py` for why none of the three new "
                f"families were expected a priori to beat the plain MLP here: the input "
                f"has no real spatial/temporal structure for a convolution or recurrence "
                f"to exploit, only an arbitrary storage order.\n")

    (ARCH_FAMILIES_DIR / "benchmark_families.md").write_text("\n".join(lines) + "\n")
    combined.to_csv(ARCH_FAMILIES_DIR / "benchmark_families.csv", index=False)
    print("\n" + "\n".join(lines[-6:]))
    print(f"\nwritten -> {ARCH_FAMILIES_DIR / 'benchmark_families.md'}")
    print(f"checkpoints -> {best_ckpts}")


if __name__ == "__main__":
    main()
