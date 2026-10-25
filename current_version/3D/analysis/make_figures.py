"""The Study II figures that go in the paper (vector PDF, one message each). Run after study2_analysis.py.
  fig_arch.pdf      accuracy vs cost for the 12 architectures (3-D and 4-D)
  fig_fit.pdf       surrogate fit: parity of both ABC features and zero-head reliability   [from select_and_report output]
  fig_post3d.pdf    joint posterior of (p1, p2, tau) for one dataset, GPS-ABC vs DNN-ABC vs NPE, truth marked
  fig_recovery.pdf  per-dataset posterior median and 95% interval of every parameter, three methods, truth line
  fig_fidelity.pdf  Wasserstein distance of DNN-ABC and NPE posteriors to the GPS-ABC posterior they approximate
  fig_sbc.pdf       SBC rank histograms (DNN-ABC, NPE)
  fig_cost.pdf      seconds per dataset, log scale
    python analysis/make_figures.py --model 4d --results results --out figures
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import study2_analysis as A

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"GPS-ABC": "#7a7a7a", "DNN-ABC": "#0b5fa5", "NPE": "#d1495b", "ABC-sim": "#2e8b57"}
LAB = {"log10 p1": r"$\log_{10}p_1$", "log10 p2": r"$\log_{10}p_2$", "tau": r"$\tau$", "log10 delta": r"$\log_{10}\delta$"}


def load_all(res, model, case):
    return {"GPS-ABC": A.load_gps(res, model, case), "DNN-ABC": A.load_npz(res, f"dnnabc_{model}_case{case}.npz", A.BURN),
            "NPE": A.load_npz(res, f"npe_{model}_case{case}.npz", 0)}


def fig_recovery(res, model, out):
    D = 3 if model == "3d" else 4; R = pd.read_csv(Path(res) / "tables" / f"recovery_raw_{model}.csv")
    fig, axs = plt.subplots(2, D, figsize=(2.9 * D, 5.2), sharex="col")
    for r, case in enumerate((1, 2)):
        tv = A.truth_vec(case, D)
        for k in range(D):
            ax = axs[r, k]; nm = A.NAMES[k]
            for j, m in enumerate(("GPS-ABC", "DNN-ABC", "NPE")):
                d = R[(R.case == case) & (R.method == m) & (R.param == nm)].sort_values("dataset").head(20)
                if d.empty: continue
                x = np.arange(len(d)) + (j - 1) * 0.25
                ax.errorbar(x, d["median"], yerr=[d["median"] - d.lo, d.hi - d["median"]], fmt="o", ms=2, lw=.7, color=COL[m], label=m if (r == 0 and k == 0) else None)
            ax.axhline(tv[k], color="k", ls="--", lw=.8); ax.set_title(f"case {case}: {LAB[nm]}", fontsize=9)
            if r == 1: ax.set_xlabel("dataset")
    axs[0, 0].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(Path(out) / f"fig_recovery_{model}.pdf"); plt.close(fig)


def fig_post3d(res, model, out, case=1, ds=1):
    M = load_all(res, model, case); tv = A.truth_vec(case, 3)
    fig = plt.figure(figsize=(11, 3.8))
    for j, m in enumerate(("GPS-ABC", "DNN-ABC", "NPE")):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        if ds in M[m]:
            s = M[m][ds][0][::max(1, len(M[m][ds][0]) // 1500), :3]; ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=2, alpha=.35, color=COL[m])
        ax.scatter(*tv, marker="*", s=140, color="gold", edgecolor="k", zorder=10)
        ax.set_xlabel(LAB["log10 p1"]); ax.set_ylabel(LAB["log10 p2"]); ax.set_zlabel(LAB["tau"]); ax.set_title(m); ax.view_init(20, -60)
        ax.set_xlim(-11, -7); ax.set_ylim(-11, -7); ax.set_zlim(0.1, 19.9)
    fig.tight_layout(); fig.savefig(Path(out) / f"fig_post3d_{model}.pdf"); plt.close(fig)


def fig_fidelity(res, model, out):
    f = Path(res) / "tables" / f"fidelity_{model}.csv"
    if not f.exists(): return
    F = pd.read_csv(f); D = 3 if model == "3d" else 4
    fig, axs = plt.subplots(1, D, figsize=(2.6 * D, 2.8))
    for k, nm in enumerate(A.NAMES[:D]):
        ax = axs[k]
        for j, m in enumerate(("DNN-ABC", "NPE")):
            d = F[(F.method == m) & (F.param == nm)]
            ax.bar(np.arange(len(d)) + j * 0.38, d["mean"], 0.36, yerr=d["std"] / np.sqrt(d["count"]), color=COL[m], label=m if k == 0 else None)
        ax.set_xticks(np.arange(2) + .19); ax.set_xticklabels(["case 1", "case 2"]); ax.set_title(LAB[nm]); ax.set_ylabel("W1 to GPS-ABC" if k == 0 else "")
    axs[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(Path(out) / f"fig_fidelity_{model}.pdf"); plt.close(fig)


def fig_sbc(res, model, out):
    f = Path(res) / f"sbc_{model}.npz"
    if not f.exists(): return
    z = np.load(f); D = 3 if model == "3d" else 4; fig, axs = plt.subplots(2, D, figsize=(2.6 * D, 4.6), sharey="row")
    for r, (key, nm, L) in enumerate((("rank_dnnabc", "DNN-ABC", int(z["L_dnnabc"])), ("rank_npe", "NPE", 1000))):
        if key not in z: continue
        for k in range(D):
            ax = axs[r, k]; h = np.histogram(z[key][:, k], bins=np.linspace(-.5, L + .5, 11))[0]
            ax.bar(range(10), h / h.sum(), color=COL[nm]); ax.axhline(.1, color="k", ls="--", lw=.7); ax.set_title(f"{nm}: {LAB[A.NAMES[k]]}", fontsize=8); ax.set_xticks([])
    fig.tight_layout(); fig.savefig(Path(out) / f"fig_sbc_{model}.pdf"); plt.close(fig)


def fig_cost(res, model, out):
    f = Path(res) / "tables" / f"recovery_{model}.csv"
    if not f.exists(): return
    T = pd.read_csv(f).groupby("method").secs.mean().reindex(["GPS-ABC", "DNN-ABC", "NPE"]).dropna()
    fig, ax = plt.subplots(figsize=(3.2, 2.6)); ax.bar(T.index, T.values, color=[COL[m] for m in T.index]); ax.set_yscale("log"); ax.set_ylabel("seconds per dataset")
    for i, v in enumerate(T.values): ax.text(i, v * 1.15, f"{v:.3g}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(Path(out) / f"fig_cost_{model}.pdf"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--results", default="results"); ap.add_argument("--out", default="figures")
    a = ap.parse_args(); Path(a.out).mkdir(exist_ok=True, parents=True)
    fig_recovery(a.results, a.model, a.out)
    if a.model == "3d": fig_post3d(a.results, a.model, a.out)
    else: fig_post3d(a.results, a.model, a.out)      # first three coordinates (p1, p2, tau) of the 4-D posterior
    fig_fidelity(a.results, a.model, a.out); fig_sbc(a.results, a.model, a.out); fig_cost(a.results, a.model, a.out)
    print("figures written to", a.out)
