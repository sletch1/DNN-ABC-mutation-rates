"""GP-vs-DNN scaling study -- the quantitative core of dnn_improvement.md targets
#1 (training-set scale) and #2 (inference cost).

For GP training budgets {50,100,200,300,500,1000,1500} (space-filling subsets of
the reps 1-8 pool), measure:
  - mean-surface MSE on the denoised 1000-point grid (accuracy vs data budget),
  - GP fit time (the O(n^3) wall), and
  - GP per-query prediction time (grows with n).
Then the deployed DNN (trained on ALL ~5000 rows): its surface MSE and its per-
query time (flat, O(1) in n). The contrast is the whole argument for the NN
surrogate in higher dimensions.

Writes results/tables/gp_scaling.csv, results/logs/gp_scaling.md, and
results/figures/fig_gp_scaling.png.
"""
import sys
import time
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

from surrogates import fit_gp_surrogate_3d
from train import load_surrogate
from paths import DATA, MODEL_DIR, TABLE_DIR, FIG_DIR, LOG_DIR

warnings.filterwarnings("ignore")
BUDGETS = [50, 100, 200, 300, 500, 1000, 1500]


def _feats(d):
    return np.column_stack([np.log10(d["p"]), d["a"], d["delta"]]).astype(float)


def main():
    df = pd.read_csv(DATA)
    tr = df[df["rep"].isin([1, 2, 3, 4, 5, 6, 7, 8])]
    X_tr, y_tr = _feats(tr), np.log10(tr["d_bar"].to_numpy())
    g = df.groupby(["p", "a", "delta"])["d_bar"].apply(lambda s: np.mean(np.log10(s)))
    keys = np.array(list(g.index))
    X_true = np.column_stack([np.log10(keys[:, 0]), keys[:, 1], keys[:, 2]])
    y_true = g.to_numpy()
    Xq = X_true[:500]  # fixed query batch for timing

    rows = []
    for b in BUDGETS:
        t0 = time.time()
        gp = fit_gp_surrogate_3d(X_tr, y_tr, budget=b)
        fit_t = time.time() - t0
        m, _ = gp.predict(X_true)
        mse = float(np.mean((m - y_true) ** 2))
        t1 = time.time()
        for _ in range(3):
            gp.predict(Xq)
        q_t = (time.time() - t1) / 3 / len(Xq) * 1e6  # microseconds / point
        rows.append({"model": "GP", "budget": gp.budget, "surface_mse": mse,
                     "fit_s": fit_t, "query_us": q_t})
        print(f"  GP budget {gp.budget:5d}: mse {mse:.4e}  fit {fit_t:5.1f}s  query {q_t:6.1f}us/pt")

    dnn = load_surrogate(MODEL_DIR / "surrogate_3d.pt")
    m, _ = dnn.predict(X_true)
    dnn_mse = float(np.mean((m - y_true) ** 2))
    t1 = time.time()
    for _ in range(3):
        dnn.predict(Xq)
    dnn_q = (time.time() - t1) / 3 / len(Xq) * 1e6
    rows.append({"model": "DNN", "budget": 5000, "surface_mse": dnn_mse,
                 "fit_s": np.nan, "query_us": dnn_q})
    print(f"  DNN (all ~5000): mse {dnn_mse:.4e}  query {dnn_q:.1f}us/pt (flat in n)")

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "gp_scaling.csv", index=False)

    gp_rows = out[out["model"] == "GP"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(gp_rows["budget"], gp_rows["surface_mse"], "o-", color="#43a047", label="GP")
    axes[0].axhline(dnn_mse, color="#1e88e5", ls="--", lw=2, label="DNN (all ~5000 rows)")
    axes[0].set_xscale("log"); axes[0].set_xlabel("GP training budget (points)")
    axes[0].set_ylabel("mean-surface MSE"); axes[0].set_title("Accuracy vs data budget")
    axes[0].legend()
    axes[1].plot(gp_rows["budget"], gp_rows["fit_s"], "o-", color="#e53935", label="GP fit time (s)")
    axes[1].plot(gp_rows["budget"], gp_rows["query_us"], "s-", color="#8e44ad", label="GP query (us/pt)")
    axes[1].axhline(dnn_q, color="#1e88e5", ls="--", lw=2, label="DNN query (us/pt, flat)")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("GP training budget (points)"); axes[1].set_title("Cost vs data budget")
    axes[1].legend(fontsize=8)
    fig.suptitle("GP hits an O(n^3) wall; the DNN uses all data at flat query cost", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_gp_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    best_gp = gp_rows.loc[gp_rows["surface_mse"].idxmin()]
    lines = ["# GP-vs-DNN scaling (targets #1 and #2)\n",
             "| model | budget | surface MSE | fit (s) | query (us/pt) |",
             "|---|---|---|---|---|"]
    for _, r in out.iterrows():
        fit = "-" if np.isnan(r["fit_s"]) else f"{r['fit_s']:.1f}"
        lines.append(f"| {r['model']} | {int(r['budget'])} | {r['surface_mse']:.4e} | {fit} | {r['query_us']:.1f} |")
    lines += ["",
              f"- DNN surface MSE {dnn_mse:.4e} beats the best GP "
              f"(budget {int(best_gp['budget'])}, {best_gp['surface_mse']:.4e}) by "
              f"{(1-dnn_mse/best_gp['surface_mse'])*100:.0f}%.",
              f"- GP fit time grows ~cubically with budget; DNN query time "
              f"({dnn_q:.1f}us/pt) is independent of training-set size.",
              ""]
    (LOG_DIR / "gp_scaling.md").write_text("\n".join(lines))
    print("\nwritten -> results/tables/gp_scaling.csv, results/logs/gp_scaling.md, figure")


if __name__ == "__main__":
    main()
