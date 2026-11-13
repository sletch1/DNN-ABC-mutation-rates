"""Train the heteroscedastic MLP surrogate for the 1-D constant-mutation-rate
case: log10(p) -> (mean, predictive variance) of log10(d_bar).

Ground truth: data/slow_data_1D.csv, 110 log-spaced p (10 reps each). Split
by replicate, no leakage: train = reps 1-6, val = 7-8 (early stop + conformal
calibration), test = 9-10. After training, the predictive sd is rescaled by a
single split-conformal factor so the 95% interval has valid empirical
coverage -- this is what makes the surrogate's uncertainty trustworthy inside
the ABC-MCMC acceptance step.

Usage: python train.py --data ./data/slow_data_1D.csv --outdir ./results
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
TRAIN_REPS, VAL_REPS, TEST_REPS = {1, 2, 3, 4, 5}, {6, 7, 8}, {9, 10}
DEFAULT_DATA = str(DATA)

# Selected by benchmark_arch.py (GELU, no BatchNorm/dropout: ~10x better fit
# than the original ReLU+BatchNorm, which was badly biased at the domain
# edges) and benchmark_capacity.py/benchmark_capacity_confirm.py (32-16 is
# 14x smaller than the original 128-64 with no measurable accuracy cost,
# confirmed at Table 1's own scale -- see results/logs/benchmark_capacity_confirm.md).
ARCH = dict(hidden_dims=(32, 16), activation="gelu", use_bn=False, dropout=0.0)


def resample_dbar_J(df, J, k_resamples=5, seed=0):
    """Reconstruct a J-culture d_bar by averaging a random J-subset of the
    100 stored per-culture values, instead of resimulating. Fixes the
    surrogate's J-blindness (README §4.3b): every row was generated at
    J=100, so a surrogate trained on the stored `d_bar` column has only ever
    seen J=100 noise and is miscalibrated at smaller J.

    `k_resamples` independent subsets per row keep the training-set size
    comparable across J despite the subsampling. Returns len(df)*k_resamples
    values, row-major (all k resamples of row 0, then row 1, ...).
    """
    d_cols = [c for c in df.columns if c.startswith("d_") and c != "d_bar"]
    D = df[d_cols].to_numpy()  # (n_rows, 100)
    n_rows, n_cultures = D.shape
    assert J <= n_cultures, f"J={J} exceeds the {n_cultures} stored cultures per row"
    rng = np.random.default_rng(seed)
    out = np.empty(n_rows * k_resamples)
    for i in range(n_rows):
        for k in range(k_resamples):
            idx = rng.choice(n_cultures, size=J, replace=False)
            out[i * k_resamples + k] = D[i, idx].mean()
    return out


def load_splits(csv_path, J=None, k_resamples=5, seed=0):
    """Read the ground-truth CSV, split by replicate into (x, y) pairs of
    log10(p)/log10(d_bar) for train/val/test. J=None uses the stored
    J=100 d_bar column as-is; J=10/50 reconstructs via `resample_dbar_J`.
    """
    df = pd.read_csv(csv_path)
    assert df["a"].nunique() == 1 and df["delta"].nunique() == 1, "expected the 1D file"
    rep = df["rep"].to_numpy()

    if J is None or J == 100:
        x = np.log10(df["p"].to_numpy())
        y = np.log10(df["d_bar"].to_numpy())
        rep_expanded = rep
    else:
        x_base = np.log10(df["p"].to_numpy())
        # Same log10 floor abc_mcmc.py uses, so an all-extinct J-subsample doesn't produce -inf.
        y_base = np.log10(np.maximum(resample_dbar_J(df, J, k_resamples, seed), 1e-6))
        x = np.repeat(x_base, k_resamples)
        y = y_base
        rep_expanded = np.repeat(rep, k_resamples)

    def subset(reps):
        m = np.isin(rep_expanded, list(reps))
        return x[m], y[m]

    return subset(TRAIN_REPS), subset(VAL_REPS), subset(TEST_REPS)


def _t(a):
    """1-D numpy array -> float32 torch column tensor, shape [n, 1]."""
    return torch.tensor(a, dtype=torch.float32).unsqueeze(1)


def train_model(x_train, y_train, x_val, y_val, epochs=800, patience=40,
                warmup=60, seed=0):
    """Fit a HeteroscedasticMLP on (x_train, y_train), early-stopping on
    (x_val, y_val). Returns the model plus the two fitted Standardizers.

    First `warmup` epochs fit the mean head alone under MSE -- under the
    joint NLL an ill-fit mean can otherwise be masked by inflating the
    predicted variance instead. Keeps the best validation-NLL weights.
    """
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
                loss = torch.nn.functional.mse_loss(mean, yb)
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
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # survives later in-place updates
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
    Uses the finite-sample-corrected quantile level ceil((n+1)(1-alpha))/n on
    normalized residuals |y-mean|/sd -- guarantees marginal coverage for
    exchangeable points, where the plain (1-alpha) quantile under-covers.
    """
    surr = DNNSurrogate(model, x_scaler, y_scaler, sd_scale=1.0)
    mean, sd = surr.predict(x_val)
    norm_resid = np.abs(y_val - mean) / np.maximum(sd, 1e-9)
    n = len(norm_resid)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    q = float(np.quantile(norm_resid, level, method="higher"))
    return q / Z_975


def evaluate(surr, x, y, label):
    """Print and return fit quality for one split: MSE/MAE on the log10
    scale, MSE back on the raw d_bar scale, and empirical coverage of the
    95% predictive interval (should land near 0.95 if calibration worked).
    """
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
    """Two diagnostic figures: the fitted curve with its calibrated 95% band
    over the data, and a test-set predicted-vs-true parity plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = splits
    xg = np.linspace(-8, -1.46, 601)
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


def run(csv_path=None, outdir=None, seed=0, J=None):
    """Load -> train -> conformally calibrate -> evaluate -> save, to
    results/model/ and results/figures/. `load_surrogate` reads the
    checkpoint back without retraining. J=None trains/saves the deployed
    J=100 model; J=10/50 trains a separate J-specific one (never overwrites
    the deployed checkpoint); diagnostic plots are only made for J=None.
    """
    csv_path = csv_path or str(DATA)
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = load_splits(csv_path, J=J, seed=seed)
    label = "J=100 (deployed)" if J is None else f"J={J}"
    print(f"[{label}] train n={len(x_tr)}  val n={len(x_va)}  test n={len(x_te)}")

    model, xs, ys = train_model(x_tr, y_tr, x_va, y_va, seed=seed)
    sd_scale = calibrate_conformal(model, xs, ys, x_va, y_va)
    print(f"[{label}] conformal sd_scale = {sd_scale:.4f}")
    surr = DNNSurrogate(model, xs, ys, sd_scale=sd_scale)

    metrics = {split: evaluate(surr, x, y, split)
               for split, (x, y) in [("train", (x_tr, y_tr)),
                                      ("val", (x_va, y_va)),
                                      ("test", (x_te, y_te))]}

    ckpt_path = MODEL_DIR / "surrogate_1d.pt" if J is None else MODEL_DIR / f"surrogate_1d_J{J}.pt"
    metrics_path = (MODEL_DIR / "surrogate_metrics.json" if J is None
                    else MODEL_DIR / f"surrogate_metrics_J{J}.json")
    if J is None:
        make_plots(surr, ((x_tr, y_tr), (x_va, y_va), (x_te, y_te)), FIG_DIR)

    torch.save({"model_state": model.state_dict(),
                "x_scaler": xs.state_dict(), "y_scaler": ys.state_dict(),
                "sd_scale": sd_scale, "hidden_dims": ARCH["hidden_dims"],
                "activation": ARCH["activation"], "use_bn": ARCH["use_bn"],
                "dropout": ARCH["dropout"],
                "input": "log10(p)", "output": "log10(d_bar)",
                "heteroscedastic": True, "source_csv": str(csv_path),
                "trained_at_J": J if J is not None else 100},
               ckpt_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{label}] saved -> {ckpt_path}, {metrics_path}")
    return surr


def load_surrogate(ckpt_path):
    """Rebuild a ready-to-use DNNSurrogate from a saved checkpoint (weights,
    both Standardizers, and the conformal sd_scale) without retraining --
    this is what the ABC scripts call. `weights_only=False` because the
    checkpoint bundles config values alongside the tensors.
    """
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
    ap.add_argument("--J", type=int, default=None,
                    help="train a J-specific surrogate instead of the deployed J=100 model")
    args = ap.parse_args()
    run(args.data, seed=args.seed, J=args.J)
