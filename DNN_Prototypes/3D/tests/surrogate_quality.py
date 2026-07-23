"""Thorough surrogate-quality tests for the deployed 3-D residual-MLP surrogate.

Writes results/logs/surrogate_quality.md, results/tables/calibration.csv,
results/tables/region_error.csv, and results/figures/fig_calibration.png +
fig_region_error.png. All on the held-out test replicates (9-10); the surrogate
never trained on them.

1. CALIBRATION RELIABILITY: for nominal coverage levels {0.5,0.8,0.9,0.95,0.99},
   the empirical fraction of test targets inside mean +- z*sd (deployed, conformal-
   scaled sd). A well-calibrated heteroscedastic surrogate tracks the diagonal.
2. PER-REGION ERROR: test MSE(log10 d_bar) binned over the (a, delta) grid -- shows
   whether accuracy is uniform across the 3-D box or degrades in any corner.
3. RESIDUAL DIAGNOSTICS: mean/std of standardized residuals (should be ~0 / ~1).
"""
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "abc", _ROOT / "figures"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from train import load_surrogate, load_splits
from paths import DATA, MODEL_DIR, TABLE_DIR, FIG_DIR, LOG_DIR

warnings.filterwarnings("ignore")
LEVELS = [0.5, 0.8, 0.9, 0.95, 0.99]


def main():
    surr = load_surrogate(MODEL_DIR / "surrogate_3d.pt")
    (_, _), (_, _), (X_te, y_te) = load_splits(DATA)
    mean, sd = surr.predict(X_te)
    resid = y_te - mean
    z = resid / np.maximum(sd, 1e-9)

    # 1. calibration reliability
    cal_rows = []
    for lv in LEVELS:
        zc = norm.ppf(0.5 + lv / 2)
        emp = float(np.mean(np.abs(z) <= zc))
        cal_rows.append({"nominal": lv, "empirical": emp})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(TABLE_DIR / "calibration.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="perfect")
    ax.plot(cal["nominal"], cal["empirical"], "o-", color="#1e88e5", lw=2, label="surrogate")
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration reliability (held-out test)"); ax.set_aspect("equal")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "fig_calibration.png", dpi=150)
    plt.close(fig)

    # 2. per-region error over (a, delta)
    df = pd.read_csv(DATA)
    a_vals = np.sort(df["a"].unique()); d_vals = np.sort(df["delta"].unique())
    err_grid = np.full((len(a_vals), len(d_vals)), np.nan)
    reg_rows = []
    for i, a in enumerate(a_vals):
        for j, d in enumerate(d_vals):
            m = np.isclose(X_te[:, 1], a) & np.isclose(X_te[:, 2], d)
            if m.sum():
                mse = float(np.mean(resid[m] ** 2))
                err_grid[i, j] = mse
                reg_rows.append({"a": a, "delta": d, "n": int(m.sum()), "mse_log": mse})
    pd.DataFrame(reg_rows).to_csv(TABLE_DIR / "region_error.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(err_grid, origin="lower", aspect="auto", cmap="viridis",
                   extent=[d_vals.min(), d_vals.max(), a_vals.min(), a_vals.max()])
    ax.set_xlabel("delta"); ax.set_ylabel("a")
    ax.set_title("Per-region surrogate error: test MSE(log10 d_bar) over (a, delta)")
    fig.colorbar(im, ax=ax, label="MSE(log10 d_bar)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_region_error.png", dpi=150)
    plt.close(fig)

    # 3. residual diagnostics
    zmean, zstd = float(np.mean(z)), float(np.std(z))
    worst = max(reg_rows, key=lambda r: r["mse_log"])
    best = min(reg_rows, key=lambda r: r["mse_log"])

    def cov_badge(lv, emp):
        return "OK" if emp >= lv - 0.03 else "LOW"

    lines = ["# Surrogate quality (held-out test set)\n",
             "## 1. Calibration reliability",
             "| nominal | empirical | status |", "|---|---|---|"]
    for _, r in cal.iterrows():
        lines.append(f"| {r['nominal']:.2f} | {r['empirical']:.3f} | "
                     f"{cov_badge(r['nominal'], r['empirical'])} |")
    lines += ["", "## 2. Per-region error over (a, delta)",
              f"- overall test MSE(log) = {float(np.mean(resid**2)):.5f}",
              f"- best region:  a={best['a']:.2f}, delta={best['delta']:.2f} -> MSE {best['mse_log']:.5f}",
              f"- worst region: a={worst['a']:.2f}, delta={worst['delta']:.2f} -> MSE {worst['mse_log']:.5f}",
              f"- worst/best ratio = {worst['mse_log']/best['mse_log']:.1f}x  (uniformity of fit)",
              "", "## 3. Residual diagnostics (standardized)",
              f"- mean(z) = {zmean:+.3f} (target 0), std(z) = {zstd:.3f} (target 1)",
              ""]
    (LOG_DIR / "surrogate_quality.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\nwritten -> results/logs/surrogate_quality.md + calibration/region tables + figures")


if __name__ == "__main__":
    main()
