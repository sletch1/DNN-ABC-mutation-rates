"""Train the heteroscedastic MLP surrogate for the 1-D constant-mutation-rate
case: log10(p) -> ( mean log10(d_bar), predictive variance ).

Ground truth: NN_ABC/data/slow_data_1D.csv (exact/slow simulator, Algorithm 2),
101 log-spaced p in log10(p) in [-8,-2], 10 replicates each (1010 rows).

Splits by replicate so every p grid point appears in every split with no leakage:
  train = reps 1-6, val = reps 7-8 (early stopping + conformal calibration),
  test = reps 9-10 (held out).

After training we calibrate the predictive std by a single conformal scale factor
so the 95% predictive interval has valid empirical coverage -- this is what makes
the surrogate's uncertainty trustworthy inside the ABC-MCMC acceptance step.

Usage:
    python train.py --data ../../data/slow_data_1D.csv --outdir ./results
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

from model import HeteroscedasticMLP, Standardizer, gaussian_nll
from surrogates import DNNSurrogate
from paths import DATA, MODEL_DIR, FIG_DIR, RESULTS

ALPHA = 0.05
Z_975 = 1.959964
# Larger calibration set (3 reps = 303 pts) makes split-conformal coverage transfer
# reliably to the test reps; weights are fit on reps 1-5, calibration/early-stop on 6-8.
TRAIN_REPS, VAL_REPS, TEST_REPS = {1, 2, 3, 4, 5}, {6, 7, 8}, {9, 10}
DEFAULT_DATA = str(DATA)

# Architecture selected by benchmark_arch.py: a smooth activation (GELU) with NO
# BatchNorm and no dropout gives the best mean-curve fit for this smooth 1-D
# response -- within ~1% of the GP (a statistical tie), vs the original
# ReLU+BatchNorm design which was ~10x worse (badly biased at the domain edges).
ARCH = dict(hidden_dims=(128, 64), activation="gelu", use_bn=False, dropout=0.0)


def load_splits(csv_path):
    df = pd.read_csv(csv_path)
    assert df["a"].nunique() == 1 and df["delta"].nunique() == 1, "expected the 1D file"
    x = np.log10(df["p"].to_numpy())
    y = np.log10(df["d_bar"].to_numpy())
    rep = df["rep"].to_numpy()

    def subset(reps):
        m = np.isin(rep, list(reps))
        return x[m], y[m]

    return subset(TRAIN_REPS), subset(VAL_REPS), subset(TEST_REPS)


def _t(a):
    return torch.tensor(a, dtype=torch.float32).unsqueeze(1)


def train_model(x_train, y_train, x_val, y_val, epochs=800, patience=40,
                warmup=60, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_scaler = Standardizer().fit(_t(x_train))
    y_scaler = Standardizer().fit(_t(y_train))
    xt_tr, yt_tr = x_scaler.transform(_t(x_train)), y_scaler.transform(_t(y_train))
    xt_va, yt_va = x_scaler.transform(_t(x_val)), y_scaler.transform(_t(y_val))

    model = HeteroscedasticMLP(**ARCH)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    loader = DataLoader(TensorDataset(xt_tr, yt_tr), batch_size=32, shuffle=True)

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


def calibrate_conformal(model, x_scaler, y_scaler, x_val, y_val):
    """Split-conformal scale so mean +- 1.96*(sd*scale) has >= 95% coverage.

    Uses the finite-sample-corrected quantile level ceil((n+1)(1-alpha))/n on the
    normalized residuals |y - mean| / sd -- this is the level that guarantees
    marginal coverage >= 1-alpha for exchangeable calibration/test points (the
    plain (1-alpha) empirical quantile slightly under-covers on small sets).
    """
    surr = DNNSurrogate(model, x_scaler, y_scaler, sd_scale=1.0)
    mean, sd = surr.predict(x_val)
    norm_resid = np.abs(y_val - mean) / np.maximum(sd, 1e-9)
    n = len(norm_resid)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    q = float(np.quantile(norm_resid, level, method="higher"))
    return q / Z_975


def evaluate(surr, x, y, label):
    mean, sd = surr.predict(x)
    mse_log = float(np.mean((mean - y) ** 2))
    mae_log = float(np.mean(np.abs(mean - y)))
    mse_raw = float(np.mean((10 ** y - 10 ** mean) ** 2))
    lower, upper = mean - Z_975 * sd, mean + Z_975 * sd
    cover = float(np.mean((y >= lower) & (y <= upper)))
    print(f"[{label}] n={len(y):4d}  MSE(log)={mse_log:.5f}  MAE(log)={mae_log:.5f}  "
          f"MSE(d_bar)={mse_raw:.3e}  95%cover={cover:.3f}")
    return {"n": len(y), "mse_log": mse_log, "mae_log": mae_log,
            "mse_raw": mse_raw, "coverage95": cover}


def make_plots(surr, splits, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = splits
    xg = np.linspace(-8, -2, 601)
    mg, sg = surr.predict(xg)
    lo, hi = mg - Z_975 * sg, mg + Z_975 * sg

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_tr, 10 ** y_tr, s=10, alpha=0.35, color="tab:gray", label="train")
    ax.scatter(x_va, 10 ** y_va, s=10, alpha=0.6, color="tab:blue", label="val")
    ax.scatter(x_te, 10 ** y_te, s=10, alpha=0.6, color="tab:orange", label="test")
    ax.plot(xg, 10 ** mg, color="red", lw=2, label="DNN mean")
    ax.fill_between(xg, 10 ** lo, 10 ** hi, color="red", alpha=0.15,
                    label="95% predictive interval (calibrated)")
    ax.set_xlabel("log10(p)"); ax.set_ylabel("d_bar = mean sqrt(X/Z)")
    ax.set_title("Heteroscedastic DNN surrogate vs. exact-simulator data (1D)")
    ax.legend(); fig.tight_layout()
    fig.savefig(Path(outdir) / "surrogate_fit.png", dpi=150); plt.close(fig)

    m_te, _ = surr.predict(x_te)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, m_te, s=15, alpha=0.7)
    lims = [min(y_te.min(), m_te.min()), max(y_te.max(), m_te.max())]
    ax.plot(lims, lims, color="red", lw=1)
    ax.set_xlabel("true log10(d_bar)"); ax.set_ylabel("predicted log10(d_bar)")
    ax.set_title("Test-set parity"); ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(Path(outdir) / "surrogate_parity.png", dpi=150); plt.close(fig)


def run(csv_path=None, outdir=None, seed=0):
    # model checkpoint + metrics -> results/model/ ; diagnostic plots -> results/figures/
    csv_path = csv_path or str(DATA)
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = load_splits(csv_path)
    print(f"train n={len(x_tr)}  val n={len(x_va)}  test n={len(x_te)}")

    model, xs, ys = train_model(x_tr, y_tr, x_va, y_va, seed=seed)
    sd_scale = calibrate_conformal(model, xs, ys, x_va, y_va)
    print(f"conformal sd_scale = {sd_scale:.4f}")
    surr = DNNSurrogate(model, xs, ys, sd_scale=sd_scale)

    metrics = {split: evaluate(surr, x, y, split)
               for split, (x, y) in [("train", (x_tr, y_tr)),
                                      ("val", (x_va, y_va)),
                                      ("test", (x_te, y_te))]}
    make_plots(surr, ((x_tr, y_tr), (x_va, y_va), (x_te, y_te)), FIG_DIR)

    torch.save({"model_state": model.state_dict(),
                "x_scaler": xs.state_dict(), "y_scaler": ys.state_dict(),
                "sd_scale": sd_scale, "hidden_dims": ARCH["hidden_dims"],
                "activation": ARCH["activation"], "use_bn": ARCH["use_bn"],
                "dropout": ARCH["dropout"],
                "input": "log10(p)", "output": "log10(d_bar)",
                "heteroscedastic": True, "source_csv": str(csv_path)},
               MODEL_DIR / "surrogate_1d.pt")
    with open(MODEL_DIR / "surrogate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved -> {MODEL_DIR/'surrogate_1d.pt'}, {MODEL_DIR/'surrogate_metrics.json'}, "
          f"plots in {FIG_DIR}")
    return surr


def load_surrogate(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = HeteroscedasticMLP(hidden_dims=tuple(ckpt["hidden_dims"]),
                               activation=ckpt.get("activation", "relu"),
                               use_bn=ckpt.get("use_bn", True),
                               dropout=ckpt.get("dropout", 0.1))
    model.load_state_dict(ckpt["model_state"]); model.eval()
    xs = Standardizer().load_state_dict(ckpt["x_scaler"])
    ys = Standardizer().load_state_dict(ckpt["y_scaler"])
    return DNNSurrogate(model, xs, ys, sd_scale=ckpt["sd_scale"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.data, seed=args.seed)
