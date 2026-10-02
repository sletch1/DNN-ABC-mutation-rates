"""Train the heteroscedastic surrogate for the 3-D two-stage model:
(log10 p1, log10 p2, tau) -> (mean, predictive sd) of log10(summary stat).

Ground truth: data/slow_data_3D.csv, a 2000-point Latin hypercube, 10
replicates each (20,000 rows, J=100, tp=10, a=1). Split by replicate: train
= reps 1-5, val = 6-8 (early stop + conformal calibration), test = 9-10.

Architecture kept small: benchmark_arch.py/benchmark_round2.py measured the
irreducible noise floor (held-out target is a 2-replicate mean, so it
carries E[sigma^2]/2 of unpredictable sampling noise, floor mse_mean=1.39e-3)
and found capacity isn't the binding constraint above ~700 parameters -- a
722-param network matches a 42,562-param one within 6%, though a linear
control is 25x the floor (so a network IS needed). 64-32 is the default:
comfortably at the floor, with headroom for later design changes; chosen for
parsimony, not speed (query cost is dominated by Python/PyTorch overhead, so
59x fewer parameters buys only ~29% less latency).

After training, a split-conformal scale factor rescales the predictive sd so
the 95% interval has valid empirical coverage -- what the ABC acceptance
step's consumed variance actually needs to be trustworthy.

Usage: python train.py [--data ../data/slow_data_3D.csv] [--seed 0]
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

from simulator import SUMMARY_ROOT
from model import build, gaussian_nll, Standardizer, FEATURES_RAW
from paths import DATA, MODEL_DIR, FIG_DIR, LOG_DIR

ALPHA = 0.05
Z_975 = 1.959964
TRAIN_REPS, VAL_REPS, TEST_REPS = {1, 2, 3, 4, 5}, {6, 7, 8}, {9, 10}

# Shape from benchmark_round2.py (smallest still at the noise floor).
# Activation from benchmark_activation_select.py: all 100 ordered pairs
# screened, ten finalists refit on 15 fresh seeds, selected on VALIDATION
# (test scored once, for the winner only). gelu->tanh is the best point
# estimate (val MSE 2.980e-04, test 1.062x floor) and both are smooth,
# dead-unit-free activations, matching the structural criteria fixed in
# advance (write-up: results/logs/benchmark_activation_select.md).
ARCH = dict(kind="mlp", hidden=(64, 32), activation=["gelu", "tanh"])

def summary_from_cultures(df, root=None):
    """Summary statistic at `root`, recomputed exactly from the stored
    per-culture d_i = sqrt(X_i/Z_i): (X_i/Z_i)^(1/root) = d_i^(2/root), so
    root=2 reproduces the stored d_bar exactly and root=4 (what the paper
    uses for the two-stage model) is mean_i sqrt(d_i) -- no re-simulation
    needed. Safe at d_i=0 and d_i=1 (map to themselves under any root),
    unlike log-based alternatives undefined at d_i=1.
    """
    if root is None:
        root = SUMMARY_ROOT
    cols = [c for c in df.columns if c.startswith("d_") and c != "d_bar"]
    if not cols:
        raise ValueError("CSV has no per-culture d_i columns; cannot re-root")
    D = df[cols].to_numpy(dtype=float)
    return np.mean(D ** (2.0 / root), axis=1)


def load_splits(csv_path):
    """Read the ground truth and split by replicate. Returns (X, y, design) per split."""
    df = pd.read_csv(csv_path)
    X = np.column_stack([np.log10(df["p1"]), np.log10(df["p2"]), df["tau"]]).astype(np.float32)
    y = np.log10(summary_from_cultures(df)).astype(np.float32)
    rep, design = df["rep"].to_numpy(), df["design"].to_numpy()
    sub = lambda r: (X[np.isin(rep, list(r))], y[np.isin(rep, list(r))], design[np.isin(rep, list(r))])
    return sub(TRAIN_REPS), sub(VAL_REPS), sub(TEST_REPS)


def _t(a, col=False):
    t = torch.tensor(np.asarray(a), dtype=torch.float32)
    return t.unsqueeze(1) if col else t


def train_model(Xtr, ytr, Xva, yva, arch=None, epochs=800, patience=50, warmup=50,
                bs=256, seed=0):
    """Fit the model. Returns (model, x_scaler, y_scaler). Loss switches from
    MSE to Gaussian NLL after `warmup` epochs -- under NLL from scratch the
    variance head can mask a badly-fit mean by inflating sigma.
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
    """Split-conformal scale so mean +- 1.96*(sd*scale) has >= 95% coverage,
    via the finite-sample-corrected level ceil((n+1)(1-alpha))/n."""
    mean, sd = surr.predict(Xva)
    r = np.abs(yva - mean) / np.maximum(sd, 1e-9)
    n = len(r)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(r, level, method="higher")) / Z_975


def evaluate(surr, X, y, design, label, var_within=None):
    """Metrics for one split; `mse_mean` averages replicates per design point.
    The irreducible floor is split-specific (E[sigma^2]/r, r=5/3/2 reps for
    train/val/test) -- scoring every split against the test floor would make
    train/val look artificially superhuman.
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
    The surface is 3-D and can't be drawn directly, so the second figure
    takes fixed-tau slices and plots the fit vs. log10(p2) per p1.
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
        ax.scatter(np.log10(near.p2), np.log10(summary_from_cultures(near)), s=5, alpha=0.15,
                   color="tab:gray", label=f"data (|tau-{tau:.0f}|<0.75)")
        for p1 in p1s:
            Xg = np.column_stack([np.full_like(grid, np.log10(p1)), grid, np.full_like(grid, tau)])
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
          f"features={FEATURES_RAW}")

    # Floor must use the SAME statistic the model trains on, or "x floor" is wrong.
    df = pd.read_csv(csv_path); yy = np.log10(summary_from_cultures(df))
    var_within = float(df.assign(y=yy).groupby("design")["y"].var(ddof=1).mean())
    n_te = df[df["rep"].isin(TEST_REPS)].groupby("design").size().mean()
    print(f"E[within-design variance] = {var_within:.3e}  ->  irreducible floor on the "
          f"{n_te:.0f}-replicate test target = {var_within/n_te:.3e}\n")

    model, xs, ys = train_model(Xtr, ytr, Xva, yva, arch=arch, seed=seed)
    surr = DNNSurrogate3D(model, xs, ys, sd_scale=1.0, raw_inputs=False)
    sd_scale = calibrate_conformal(surr, Xva, yva)
    surr.sd_scale = sd_scale
    print(f"conformal sd_scale = {sd_scale:.4f}")

    # Mirrored here (not just in the .pt/simulator.py) so metrics.json is readable standalone.
    metrics = {"var_within": var_within,
               "summary_root": SUMMARY_ROOT,
               "sd_scale": float(sd_scale),
               "arch": {k: list(v) if isinstance(v, tuple) else v for k, v in arch.items()}}
    for lbl, (X, y, d) in [("train", (Xtr, ytr, dtr)), ("val", (Xva, yva, dva)),
                           ("test", (Xte, yte, dte))]:
        metrics[lbl] = evaluate(surr, X, y, d, lbl, var_within=var_within)

    make_plots(surr, ((Xtr, ytr, dtr), (Xva, yva, dva), (Xte, yte, dte)), csv_path, FIG_DIR)

    torch.save({"model_state": model.state_dict(), "x_scaler": xs.state_dict(),
                "y_scaler": ys.state_dict(), "sd_scale": sd_scale, "arch": arch,
                "input": "[log10(p1), log10(p2), tau]",
                "output": f"log10(mean_i (X_i/Z_i)^(1/{SUMMARY_ROOT}))",
                "summary_root": SUMMARY_ROOT,
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
    model = build(in_dim=3, **arch)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return DNNSurrogate3D(model,
                          Standardizer().load_state_dict(ckpt["x_scaler"]),
                          Standardizer().load_state_dict(ckpt["y_scaler"]),
                          sd_scale=ckpt["sd_scale"],
                          raw_inputs=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.data, seed=args.seed)
