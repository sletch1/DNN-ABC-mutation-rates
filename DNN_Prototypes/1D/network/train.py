"""Train the heteroscedastic MLP surrogate for the 1-D constant-mutation-rate
case: log10(p) -> ( mean log10(d_bar), predictive variance ).

Ground truth: DNN_Prototypes/1D/data/slow_data_1D.csv (exact/slow simulator, Algorithm 2),
101 log-spaced p in log10(p) in [-8,-2], 10 replicates each (1010 rows).

Splits by replicate so every p grid point appears in every split with no leakage:
  train = reps 1-6, val = reps 7-8 (early stopping + conformal calibration),
  test = reps 9-10 (held out).

After training we calibrate the predictive std by a single conformal scale factor
so the 95% predictive interval has valid empirical coverage -- this is what makes
the surrogate's uncertainty trustworthy inside the ABC-MCMC acceptance step.

Usage:
    python train.py --data ./data/slow_data_1D.csv --outdir ./results
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
    """Read the ground-truth CSV and split it into train/val/test by
    replicate (see TRAIN_REPS/VAL_REPS/TEST_REPS above). Returns
    `((x_train, y_train), (x_val, y_val), (x_test, y_test))`, where each
    x/y is a 1-D numpy array of, respectively, log10(p) and log10(d_bar).
    The assert guards against accidentally pointing this at the 3-D data
    file, which has more than one distinct value of a/delta.
    """
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
    """1-D numpy array -> float32 torch column tensor, shape [n, 1].
    The trailing `.unsqueeze(1)` adds that second dimension because the
    network (and PyTorch layers generally) expect input shaped
    [n_samples, n_features] even when n_features is 1, not a flat [n]
    vector.
    """
    return torch.tensor(a, dtype=torch.float32).unsqueeze(1)


def train_model(x_train, y_train, x_val, y_val, epochs=800, patience=40,
                warmup=60, seed=0):
    """Fit a HeteroscedasticMLP on (x_train, y_train), using x_val/y_val
    for early stopping, and return `(model, x_scaler, y_scaler)` — the
    trained network plus the two Standardizers needed to map raw inputs in
    and raw predictions back out.

    Training loop, for reference (this is the standard PyTorch pattern
    used throughout this codebase, spelled out once here):
      - `opt.zero_grad()` clears gradients left over from the previous
        batch (PyTorch accumulates gradients by default, so this is
        required every iteration).
      - `model(xb)` runs the forward pass (predictions).
      - `loss.backward()` runs backpropagation, computing d(loss)/d(param)
        for every learnable parameter via autograd.
      - `opt.step()` uses those gradients to actually update the weights
        (Adam here — a gradient-descent variant with per-parameter
        adaptive step sizes).
      - `model.train()` / `model.eval()` toggle layers that behave
        differently in training vs. inference (Dropout, BatchNorm); this
        model doesn't use either by default, but the calls are kept so the
        function is correct if `use_bn`/`dropout` are turned on.
      - `torch.no_grad()` during validation skips building the
        autograd graph, since no `.backward()` call follows — pure
        speed/memory, doesn't change the numbers.

    Two training-scheme details specific to this model:
      - **Warmup** (`epoch < warmup`): the first `warmup` epochs train the
        mean head only, on plain MSE loss. Gaussian NLL (see
        `gaussian_nll` in model.py) jointly fits mean and variance, and
        early on — when the mean is still a poor fit — the variance head
        can "cheat" by inflating predicted variance to make bad mean
        predictions look more likely under the loss, rather than the mean
        head actually improving. Warming up on MSE alone gives the mean a
        head start before the variance head is allowed to interact with
        it.
      - **Early stopping**: track the best validation NLL seen so far
        (`best_val`) and how many epochs it's been since it improved
        (`since`); if that streak reaches `patience`, stop training and
        restore the weights from the best epoch (`best_state`) rather than
        whatever the model looks like at the final epoch, which may have
        overfit past that point.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_scaler = Standardizer().fit(_t(x_train))
    y_scaler = Standardizer().fit(_t(y_train))
    xt_tr, yt_tr = x_scaler.transform(_t(x_train)), y_scaler.transform(_t(y_train))
    xt_va, yt_va = x_scaler.transform(_t(x_val)), y_scaler.transform(_t(y_val))

    model = HeteroscedasticMLP(**ARCH)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Halves the learning rate if validation loss hasn't improved for 15
    # epochs -- lets training take large steps early and fine-tune later
    # without hand-picking a decay schedule.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    # DataLoader/TensorDataset: wraps the training tensors so iterating
    # over `loader` yields shuffled mini-batches of size 32 each epoch,
    # rather than manually slicing and shuffling indices by hand.
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
            # .clone() each tensor so this snapshot survives later training
            # steps that mutate the model's live parameters in place.
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
    """Compute and print held-out performance for one split (`label` is
    just a name for the printout, e.g. "train"/"val"/"test"). Reports:
    MSE and MAE on the log10(d_bar) scale the model is actually trained
    on, MSE back on the raw d_bar scale (10**y), and empirical coverage of
    the 95% predictive interval (should be close to 0.95 if the
    conformal-calibrated uncertainty is trustworthy). Returns the same
    numbers as a dict for JSON logging.
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
    """Save two diagnostic figures to `outdir`: (1) the fitted mean curve
    and calibrated 95% band over the full log10(p) range, with the raw
    train/val/test points overlaid, and (2) a predicted-vs-true parity
    plot on the test split (points should sit on the y=x line if the
    surrogate is accurate)."""
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
    """End-to-end entry point: load data -> train -> conformally calibrate
    -> evaluate on all three splits -> save plots, model checkpoint, and a
    metrics JSON. This is what `python train.py` (see `__main__` below)
    ultimately calls; `load_surrogate` is the matching function that reads
    the checkpoint this writes, for use elsewhere (e.g. inside the ABC
    sampler) without retraining. Model checkpoint + metrics land in
    results/model/; diagnostic plots land in results/figures/.
    """
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
    """Rebuild a DNNSurrogate from a checkpoint saved by `run` above —
    reconstructs the network with the same architecture settings, loads
    its trained weights, restores the two Standardizers, and wraps
    everything in a DNNSurrogate with the saved conformal `sd_scale`. This
    is the function other scripts (e.g. the ABC sampler) should call to
    get a ready-to-use surrogate without retraining.
    `weights_only=False` is needed because the checkpoint bundles plain
    Python objects (the Standardizer state dicts, config values) alongside
    the tensor weights; PyTorch's stricter `weights_only=True` loading
    mode only accepts tensors.
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
    args = ap.parse_args()
    run(args.data, seed=args.seed)
