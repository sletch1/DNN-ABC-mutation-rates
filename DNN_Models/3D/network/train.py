"""Train the heteroscedastic surrogate for the 3-D two-stage model.

    (log10 p1, log10 p2, tau)  ->  ( mean log10(d_bar), predictive sd )

Ground truth: data/slow_data_3D.csv -- the exact cell-by-cell two-stage
simulator over a 2000-point Latin hypercube in (log10 p1, log10 p2, tau) with 10
replicates each (20,000 rows, J = 100, tp = 10, a = 1).

SPLIT, by replicate, so every design point appears in every split and no
parameter combination leaks across them:
    train = reps 1-5 (10,000 rows)   fit the weights
    val   = reps 6-8 ( 6,000 rows)   early stopping + conformal calibration
    test  = reps 9-10 ( 4,000 rows)  held out, reported only

WHY THE ARCHITECTURE IS SMALL. The architecture search
(architecture_search/benchmark_arch.py, then benchmark_round2.py) measured the
data's IRREDUCIBLE NOISE FLOOR: because the held-out target is itself a
2-replicate mean, it carries E[sigma^2]/2 of sampling noise that no model can
predict away. That floor is mse_mean = 1.39e-3, i.e. a maximum achievable
R^2 of 0.99528. Measured against it (2 seeds each):

    256-128-64  42,562 par   1.08x floor   R^2 0.99491   56 us/query
    128-64       9,026 par   1.12x floor   R^2 0.99472   43 us/query
    64-32        2,466 par   1.13x floor   R^2 0.99467   41 us/query
    32-16          722 par   1.14x floor   R^2 0.99460   40 us/query
    8-4             86 par   1.23x floor   R^2 0.99419   39 us/query
    linear          10 par  24.77x floor   R^2 0.88300   29 us/query

Two things follow. First, a network IS needed: the linear control is 25x the
floor. Second, capacity is NOT the binding constraint anywhere above ~700
parameters -- a 722-parameter network matches a 42,562-parameter one to within
6%, and every one of them is within 25% of a perfect model. Activation choice,
LayerNorm, and residual depth were likewise ties in round 1.

So the model is chosen small, but the honest reason is parsimony rather than
speed: query latency here is dominated by Python/PyTorch call overhead, not by
arithmetic, so 59x fewer parameters buys only ~29% less latency (56 -> 40 us).
64-32 is taken as the default -- comfortably at the floor, with some headroom if
the design is later widened (bigger parameter ranges, or a 4-D extension with
delta) without becoming an oversized model for the surface it fits.

CALIBRATION. After training, a single split-conformal scale factor rescales the
predictive sd so the 95% interval has valid empirical coverage on held-out data.
This is what makes the surrogate's uncertainty trustworthy inside the ABC
acceptance step, which consumes that variance directly.

Usage:
    python train.py                       # uses paths.DATA
    python train.py --data ../data/slow_data_3D.csv --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc", _ROOT / "figures"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd
import torch

from model import build, gaussian_nll, Standardizer, add_derived, FEATURES_ALL
from paths import DATA, MODEL_DIR, FIG_DIR, LOG_DIR

ALPHA = 0.05
Z_975 = 1.959964
TRAIN_REPS, VAL_REPS, TEST_REPS = {1, 2, 3, 4, 5}, {6, 7, 8}, {9, 10}

# Selected by architecture_search/benchmark_round2.py: the smallest hidden shape
# still at the noise floor. Widen only if the design or the parameter ranges
# change enough to make the surface harder.
ARCH = dict(kind="mlp", hidden=(64, 32), activation="gelu")
USE_DERIVED = True   # append log10(p_eff); see model.py for the derivation


def load_splits(csv_path, use_derived=USE_DERIVED):
    """Read the ground truth and split by replicate. Returns (X, y, design) per split."""
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


def train_model(Xtr, ytr, Xva, yva, arch=None, epochs=800, patience=50, warmup=50,
                bs=256, seed=0):
    """Fit the model. Returns (model, x_scaler, y_scaler).

    The loss switches from plain MSE to Gaussian NLL after `warmup` epochs.
    Training both heads from scratch under NLL lets the variance head explain a
    badly-fit mean by inflating sigma; settling the mean first avoids that.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    arch = arch or ARCH
    xs = Standardizer().fit(_t(Xtr)); ys = Standardizer().fit(_t(ytr, col=True))
    xt, yt = xs.transform(_t(Xtr)), ys.transform(_t(ytr, col=True))
    xv, yv = xs.transform(_t(Xva)), ys.transform(_t(yva, col=True))

    model = build(in_dim=Xtr.shape[1], **arch)
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
            loss.backward(); opt.step()
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
                print(f"early stop at epoch {ep} (best val NLL={best:.4f})")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model.eval(), xs, ys


def calibrate_conformal(surr, Xva, yva):
    """Split-conformal scale so mean +- 1.96*(sd*scale) has >= 95% coverage.

    Uses the finite-sample-corrected level ceil((n+1)(1-alpha))/n on the
    normalized residuals |y - mean| / sd, which guarantees marginal coverage
    >= 1-alpha for exchangeable calibration/test points.
    """
    mean, sd = surr.predict(Xva)
    r = np.abs(yva - mean) / np.maximum(sd, 1e-9)
    n = len(r)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(r, level, method="higher")) / Z_975


def evaluate(surr, X, y, design, label, var_within=None):
    """Metrics for one split; `mse_mean` averages replicates per design point.

    The irreducible floor is SPLIT-SPECIFIC: it is E[sigma^2]/r where r is the
    number of replicates that split contributes per design point (5 / 3 / 2 for
    train / val / test). Scoring every split against the test floor would make
    train and val look artificially superhuman, so each is compared to its own.
    """
    mean, sd = surr.predict(X)
    g = pd.DataFrame({"design": design, "y": y, "m": mean}).groupby("design").mean()
    out = dict(n=len(y),
               mse_mean=float(np.mean((g.m - g.y) ** 2)),
               mse_obs=float(np.mean((mean - y) ** 2)),
               mae=float(np.mean(np.abs(mean - y))),
               coverage95=float(np.mean(np.abs(y - mean) <= Z_975 * sd)))
    if var_within:
        reps = len(y) / max(len(np.unique(design)), 1)
        out["reps_per_design"] = reps
        out["floor"] = var_within / reps
        out["x_floor"] = out["mse_mean"] / out["floor"]
    print(f"[{label:5s}] n={out['n']:5d}  mse_mean={out['mse_mean']:.3e}"
          + (f" ({out['x_floor']:.2f}x its {out['reps_per_design']:.0f}-rep floor)"
             if "x_floor" in out else "")
          + f"  mse_obs={out['mse_obs']:.3e}  95%cover={out['coverage95']:.3f}")
    return out


def make_plots(surr, splits, csv_path, outdir):
    """Parity on the held-out test set, plus slices of the fitted surface.

    The surface is 3-D and cannot be drawn directly, so the second figure takes
    fixed-tau slices and, within each, plots the fit against log10(p2) for a few
    values of p1 -- reading the surface one axis at a time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (_, _, _), (_, _, _), (Xte, yte, dte) = splits
    m, _ = surr.predict(Xte)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yte, m, s=6, alpha=0.25, color="tab:blue")
    lims = [min(yte.min(), m.min()), max(yte.max(), m.max())]
    ax.plot(lims, lims, color="red", lw=1)
    ax.set_xlabel("true log10(d_bar)"); ax.set_ylabel("predicted log10(d_bar)")
    ax.set_title("3-D two-stage surrogate: held-out parity"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(Path(outdir) / "surrogate_parity.png", dpi=150)
    plt.close(fig)

    df = pd.read_csv(csv_path)
    tp = float(df["tp"].iloc[0])
    taus = [2.0, 5.0, 8.0]
    p1s = [1e-5, 1e-3, 3e-2]
    grid = np.linspace(-5, -1.3, 120)
    fig, axes = plt.subplots(1, len(taus), figsize=(4.6 * len(taus), 3.6), sharey=True)
    for j, tau in enumerate(taus):
        ax = axes[j]
        near = df[np.abs(df.tau - tau) < 0.75]
        ax.scatter(np.log10(near.p2), np.log10(near.d_bar), s=5, alpha=0.15,
                   color="tab:gray", label=f"data (|tau-{tau:.0f}|<0.75)")
        for p1 in p1s:
            Xg = np.column_stack([np.full_like(grid, np.log10(p1)), grid, np.full_like(grid, tau)])
            if USE_DERIVED:
                Xg = add_derived(Xg, tp=tp)
            mg, sg = surr.predict(Xg)
            ax.plot(grid, mg, lw=2, label=f"p1={p1:.0e}")
            ax.fill_between(grid, mg - Z_975 * sg, mg + Z_975 * sg, alpha=0.15)
        ax.set_title(f"tau = {tau:.0f}"); ax.set_xlabel("log10(p2)")
        if j == 0:
            ax.set_ylabel("log10(d_bar)"); ax.legend(fontsize=7)
    fig.suptitle("Fitted surface sliced by tau, with calibrated 95% bands")
    fig.tight_layout(); fig.savefig(Path(outdir) / "surrogate_fit.png", dpi=150,
                                    bbox_inches="tight")
    plt.close(fig)


def run(csv_path=None, seed=0, arch=None):
    from surrogates import DNNSurrogate3D           # local import: avoids a cycle

    csv_path = csv_path or str(DATA)
    arch = arch or ARCH
    (Xtr, ytr, dtr), (Xva, yva, dva), (Xte, yte, dte) = load_splits(csv_path)
    print(f"train n={len(ytr)}  val n={len(yva)}  test n={len(yte)}  "
          f"features={FEATURES_ALL if USE_DERIVED else FEATURES_ALL[:3]}")

    df = pd.read_csv(csv_path); yy = np.log10(df["d_bar"])
    var_within = float(df.assign(y=yy).groupby("design")["y"].var(ddof=1).mean())
    n_te = df[df["rep"].isin(TEST_REPS)].groupby("design").size().mean()
    print(f"E[within-design variance] = {var_within:.3e}  ->  irreducible floor on the "
          f"{n_te:.0f}-replicate test target = {var_within/n_te:.3e}\n")

    model, xs, ys = train_model(Xtr, ytr, Xva, yva, arch=arch, seed=seed)
    surr = DNNSurrogate3D(model, xs, ys, sd_scale=1.0, use_derived=USE_DERIVED, raw_inputs=False)
    sd_scale = calibrate_conformal(surr, Xva, yva)
    surr.sd_scale = sd_scale
    print(f"conformal sd_scale = {sd_scale:.4f}")

    metrics = {"var_within": var_within,
               "arch": {k: list(v) if isinstance(v, tuple) else v for k, v in arch.items()}}
    for lbl, (X, y, d) in [("train", (Xtr, ytr, dtr)), ("val", (Xva, yva, dva)),
                           ("test", (Xte, yte, dte))]:
        metrics[lbl] = evaluate(surr, X, y, d, lbl, var_within=var_within)

    make_plots(surr, ((Xtr, ytr, dtr), (Xva, yva, dva), (Xte, yte, dte)), csv_path, FIG_DIR)

    torch.save({"model_state": model.state_dict(), "x_scaler": xs.state_dict(),
                "y_scaler": ys.state_dict(), "sd_scale": sd_scale, "arch": arch,
                "use_derived": USE_DERIVED,
                "input": "[log10(p1), log10(p2), tau]", "output": "log10(d_bar)",
                "heteroscedastic": True, "source_csv": str(csv_path)},
               MODEL_DIR / "surrogate_3d.pt")
    (MODEL_DIR / "surrogate_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved -> {MODEL_DIR/'surrogate_3d.pt'}, {MODEL_DIR/'surrogate_metrics.json'}, "
          f"plots in {FIG_DIR}")
    return surr


def load_surrogate(ckpt_path):
    """Rebuild a DNNSurrogate3D from a checkpoint written by `run`."""
    from surrogates import DNNSurrogate3D
    ckpt = torch.load(ckpt_path, weights_only=False)
    arch = dict(ckpt["arch"])
    in_dim = 4 if ckpt.get("use_derived", True) else 3
    model = build(in_dim=in_dim, **arch)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return DNNSurrogate3D(model,
                          Standardizer().load_state_dict(ckpt["x_scaler"]),
                          Standardizer().load_state_dict(ckpt["y_scaler"]),
                          sd_scale=ckpt["sd_scale"],
                          use_derived=ckpt.get("use_derived", True),
                          raw_inputs=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.data, seed=args.seed)
