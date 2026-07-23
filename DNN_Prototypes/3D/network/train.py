"""Train the heteroscedastic residual-MLP surrogate for the 3-D constant-rate
case: (log10(p), a, delta) -> ( mean log10(d_bar), predictive variance ).

Ground truth: 3D/data/slow_data_3D.csv (exact/slow simulator, Algorithm 2),
a 10x10x10 grid of (p, a, delta) with log10(p) in [-8,-2], a, delta in [0.5,2],
10 replicates each (~10000 rows, J=100).

Splits by replicate so every (p, a, delta) grid point appears in every split with
no leakage:
  train = reps 1-5, val = reps 6-8 (early stopping + conformal calibration),
  test  = reps 9-10 (held out).

After training we calibrate the predictive std by a single conformal scale factor
so the 95% predictive interval has valid empirical coverage -- this is what makes
the surrogate's uncertainty trustworthy inside the ABC-MCMC acceptance step.

Usage:
    python train.py                 # uses paths.DATA
    python train.py --data ../data/slow_data_3D.csv --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

# --- make sibling code folders + paths.py importable (package uses flat imports) ---
_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc", _ROOT / "figures"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import HeteroscedasticResMLP, Standardizer, gaussian_nll
from surrogates import DNNSurrogate3D
from paths import DATA, MODEL_DIR, FIG_DIR

ALPHA = 0.05
Z_975 = 1.959964
TRAIN_REPS, VAL_REPS, TEST_REPS = {1, 2, 3, 4, 5}, {6, 7, 8}, {9, 10}
DEFAULT_DATA = str(DATA)

# Architecture selected by architecture_search/benchmark_arch.py: a 3-block
# residual MLP (width 128) with SiLU + LayerNorm. Residual depth + LayerNorm
# (never BatchNorm -- see the 1-D lesson) fit the (p, a, delta) interaction
# surface where the shallow 1-D funnel MLP and a budget-limited GP both fall short.
ARCH = dict(width=128, n_blocks=3, activation="silu", use_ln=True, dropout=0.0)

FEATURES = ["log10p", "a", "delta"]


def load_splits(csv_path):
    df = pd.read_csv(csv_path)
    X = np.column_stack([np.log10(df["p"].to_numpy()),
                         df["a"].to_numpy(),
                         df["delta"].to_numpy()]).astype(np.float32)
    y = np.log10(df["d_bar"].to_numpy()).astype(np.float32)
    rep = df["rep"].to_numpy()

    def subset(reps):
        m = np.isin(rep, list(reps))
        return X[m], y[m]

    return subset(TRAIN_REPS), subset(VAL_REPS), subset(TEST_REPS)


def _tx(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def _ty(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32).unsqueeze(1)


def train_model(X_train, y_train, X_val, y_val, arch=None, epochs=1200,
                patience=60, warmup=80, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    arch = arch or ARCH

    x_scaler = Standardizer().fit(_tx(X_train))
    y_scaler = Standardizer().fit(_ty(y_train))
    xt_tr, yt_tr = x_scaler.transform(_tx(X_train)), y_scaler.transform(_ty(y_train))
    xt_va, yt_va = x_scaler.transform(_tx(X_val)), y_scaler.transform(_ty(y_val))

    model = HeteroscedasticResMLP(in_dim=X_train.shape[1], **arch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    loader = DataLoader(TensorDataset(xt_tr, yt_tr), batch_size=128, shuffle=True)

    best_val, best_state, since = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            mean, logvar = model(xb)
            if epoch < warmup:
                loss = torch.nn.functional.mse_loss(mean, yb)  # stabilize the mean head first
            else:
                loss = gaussian_nll(mean, logvar, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            mean_v, logvar_v = model(xt_va)
            val_loss = gaussian_nll(mean_v, logvar_v, yt_va).item()
        sched.step(val_loss)

        if epoch >= warmup and val_loss < best_val - 1e-5:
            best_val, since = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch >= warmup:
            since += 1
            if since >= patience:
                print(f"Early stopping at epoch {epoch} (best val NLL={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, x_scaler, y_scaler


def calibrate_conformal(model, x_scaler, y_scaler, X_val, y_val):
    """Split-conformal scale so mean +- 1.96*(sd*scale) has >= 95% coverage.

    Uses the finite-sample-corrected quantile level ceil((n+1)(1-alpha))/n on the
    normalized residuals |y - mean| / sd, guaranteeing marginal coverage >= 1-alpha
    for exchangeable calibration/test points.
    """
    surr = DNNSurrogate3D(model, x_scaler, y_scaler, sd_scale=1.0)
    mean, sd = surr.predict(X_val)
    norm_resid = np.abs(y_val - mean) / np.maximum(sd, 1e-9)
    n = len(norm_resid)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    q = float(np.quantile(norm_resid, level, method="higher"))
    return q / Z_975


def evaluate(surr, X, y, label):
    mean, sd = surr.predict(X)
    mse_log = float(np.mean((mean - y) ** 2))
    mae_log = float(np.mean(np.abs(mean - y)))
    mse_raw = float(np.mean((10 ** y - 10 ** mean) ** 2))
    lower, upper = mean - Z_975 * sd, mean + Z_975 * sd
    cover = float(np.mean((y >= lower) & (y <= upper)))
    print(f"[{label}] n={len(y):4d}  MSE(log)={mse_log:.5f}  MAE(log)={mae_log:.5f}  "
          f"MSE(d_bar)={mse_raw:.3e}  95%cover={cover:.3f}")
    return {"n": len(y), "mse_log": mse_log, "mae_log": mae_log,
            "mse_raw": mse_raw, "coverage95": cover}


def make_plots(surr, splits, csv_path, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits

    # --- parity on the held-out test set ---
    m_te, _ = surr.predict(X_te)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, m_te, s=8, alpha=0.4, color="tab:blue")
    lims = [min(y_te.min(), m_te.min()), max(y_te.max(), m_te.max())]
    ax.plot(lims, lims, color="red", lw=1)
    ax.set_xlabel("true log10(d_bar)"); ax.set_ylabel("predicted log10(d_bar)")
    ax.set_title("3-D surrogate: held-out test parity"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(Path(outdir) / "surrogate_parity.png", dpi=150)
    plt.close(fig)

    # --- fit along log10(p) for representative (a, delta) slices ---
    df = pd.read_csv(csv_path)
    a_vals = np.sort(df["a"].unique()); d_vals = np.sort(df["delta"].unique())
    a_pick = [a_vals[1], a_vals[4], a_vals[8]]      # ~0.67, 1.17, 1.83
    d_pick = [d_vals[1], d_vals[4], d_vals[8]]
    pg = np.linspace(-8, -2, 200)
    fig, axes = plt.subplots(len(a_pick), len(d_pick),
                             figsize=(4.0 * len(d_pick), 3.2 * len(a_pick)),
                             sharex=True)
    for i, a in enumerate(a_pick):
        for j, d in enumerate(d_pick):
            ax = axes[i, j]
            sub = df[(np.isclose(df["a"], a)) & (np.isclose(df["delta"], d))]
            ax.scatter(np.log10(sub["p"]), np.log10(sub["d_bar"]),
                       s=10, alpha=0.4, color="tab:gray", label="data (all reps)")
            Xg = np.column_stack([pg, np.full_like(pg, a), np.full_like(pg, d)])
            mg, sg = surr.predict(Xg)
            ax.plot(pg, mg, color="red", lw=2, label="DNN mean")
            ax.fill_between(pg, mg - Z_975 * sg, mg + Z_975 * sg,
                            color="red", alpha=0.15)
            ax.set_title(f"a={a:.2f}, delta={d:.2f}", fontsize=9)
            if i == len(a_pick) - 1:
                ax.set_xlabel("log10(p)")
            if j == 0:
                ax.set_ylabel("log10(d_bar)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("3-D surrogate fit along log10(p) at fixed (a, delta) slices", y=1.0)
    fig.tight_layout(); fig.savefig(Path(outdir) / "surrogate_fit.png", dpi=150,
                                    bbox_inches="tight")
    plt.close(fig)


def run(csv_path=None, outdir=None, seed=0, arch=None):
    csv_path = csv_path or str(DATA)
    arch = arch or ARCH
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_splits(csv_path)
    print(f"train n={len(X_tr)}  val n={len(X_va)}  test n={len(X_te)}  (features={FEATURES})")

    model, xs, ys = train_model(X_tr, y_tr, X_va, y_va, arch=arch, seed=seed)
    sd_scale = calibrate_conformal(model, xs, ys, X_va, y_va)
    print(f"conformal sd_scale = {sd_scale:.4f}")
    surr = DNNSurrogate3D(model, xs, ys, sd_scale=sd_scale)

    metrics = {split: evaluate(surr, X, y, split)
               for split, (X, y) in [("train", (X_tr, y_tr)),
                                      ("val", (X_va, y_va)),
                                      ("test", (X_te, y_te))]}
    make_plots(surr, ((X_tr, y_tr), (X_va, y_va), (X_te, y_te)), csv_path, FIG_DIR)

    torch.save({"model_state": model.state_dict(),
                "x_scaler": xs.state_dict(), "y_scaler": ys.state_dict(),
                "sd_scale": sd_scale, "arch": arch,
                "input": "[log10(p), a, delta]", "output": "log10(d_bar)",
                "heteroscedastic": True, "source_csv": str(csv_path)},
               MODEL_DIR / "surrogate_3d.pt")
    with open(MODEL_DIR / "surrogate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved -> {MODEL_DIR/'surrogate_3d.pt'}, {MODEL_DIR/'surrogate_metrics.json'}, "
          f"plots in {FIG_DIR}")
    return surr


def load_surrogate(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = HeteroscedasticResMLP(in_dim=3, **ckpt["arch"])
    model.load_state_dict(ckpt["model_state"]); model.eval()
    xs = Standardizer().load_state_dict(ckpt["x_scaler"])
    ys = Standardizer().load_state_dict(ckpt["y_scaler"])
    return DNNSurrogate3D(model, xs, ys, sd_scale=ckpt["sd_scale"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.data, seed=args.seed)
