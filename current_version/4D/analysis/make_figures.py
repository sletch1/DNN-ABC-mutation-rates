"""The Study II figures that go in the paper (vector PDF, one message each). Run after study2_analysis.py.
  fig_arch.pdf      accuracy vs cost for the 12 architectures (3-D and 4-D)
  fig_fit.pdf       surrogate fit: parity of both ABC features and zero-head reliability   [from select_and_report output]
  fig_post3d.pdf    joint posterior of (p1, p2, tau) for one dataset, GPS-ABC vs DNN-ABC vs NPE, truth marked
  fig_recovery.pdf  per-dataset posterior median and 95% interval of every parameter, three methods, truth line
  fig_fidelity.pdf  Wasserstein distance of DNN-ABC and NPE posteriors to the GPS-ABC posterior they approximate
  fig_sbc.pdf       SBC rank histograms (DNN-ABC, NPE)
    python analysis/make_figures.py --model 4d --results results --out figures
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import study2_analysis as A

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"GPS-ABC": "#7a7a7a", "DNN-ABC": "#0b5fa5", "DNN-ABC (1,500)": "#7fb2d9", "DNN-ABC-rho": "#5b3fa0", "DNN-ABC-rho (1,500)": "#b3a3d6", "NPE": "#d1495b", "ABC-sim": "#2e8b57"}
LAB = {"log10 p1": r"$\log_{10}p_1$", "log10 p2": r"$\log_{10}p_2$", "tau": r"$\tau$", "log10 delta": r"$\log_{10}\delta$"}


HORIZON = None


def load_all(res, model, case):
    return {"GPS-ABC": A.load_gps(res, model, case, HORIZON), "DNN-ABC": A.load_npz(res, f"dnnabc_{model}_case{case}.npz", A.BURN_DNN), "DNN-ABC-rho": A.load_npz(res, f"dnnabcrho_{model}_case{case}.npz", A.BURN_DNN),
            "NPE": A.load_npz(res, f"npe_{model}_case{case}.npz", 0)}


PRIOR_W = {"log10 p1": 4.0, "log10 p2": 4.0, "tau": 19.8, "log10 delta": float(np.log10(2) - np.log10(.5))}


def fig_recovery(res, model, out):
    """(top) how much of the prior range each method's 95% interval spans, per parameter; (bottom) intervals of the two
    identified parameters (log10 p2, log10 delta) for the first 20 datasets, truth dashed."""
    D = 3 if model == "3d" else 4; R = pd.read_csv(Path(res) / "tables" / f"recovery_raw_{model}.csv")
    meths = [m for m in ("GPS-ABC", "DNN-ABC", "DNN-ABC-rho", "NPE") if m in set(R.method)]
    fig = plt.figure(figsize=(11, 6.4)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25])
    ax = fig.add_subplot(gs[0, :]); names = A.NAMES[:D]; w = 0.8 / len(meths)
    for j, m in enumerate(meths):
        v = [(R[(R.method == m) & (R.param == nm)].eval("(hi - lo)") / PRIOR_W[nm]) for nm in names]
        ax.bar(np.arange(D) + (j - (len(meths) - 1) / 2) * w, [x.mean() for x in v], w * .92, yerr=[x.std() / np.sqrt(len(x)) for x in v], color=COL[m], label=m, capsize=2)
    ax.set_xticks(range(D)); ax.set_xticklabels([LAB[n] for n in names]); ax.set_ylim(0, 1.05); ax.axhline(1, color="k", lw=.6, ls=":")
    ax.set_ylabel("95% interval / prior range"); ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.18))
    for c, case in enumerate((1, 2)):
        tv = A.truth_vec(case, D)
        for k, nm in enumerate(("log10 p2", "log10 delta")):
            if nm not in names: continue
            a2 = fig.add_subplot(gs[1, c * 2 + k]); kk = names.index(nm)
            for j, m in enumerate(meths):
                d = R[(R.case == case) & (R.method == m) & (R.param == nm)].sort_values("dataset").head(20)
                if d.empty: continue
                a2.errorbar(np.arange(len(d)) + (j - (len(meths) - 1) / 2) * .25, d["median"], yerr=[d["median"] - d.lo, d.hi - d["median"]], fmt="o", ms=2.2, lw=.8, color=COL[m])
            a2.axhline(tv[kk], color="k", ls="--", lw=.8); a2.set_title(f"case {case}: {LAB[nm]}", fontsize=9); a2.set_xlabel("dataset")
    fig.tight_layout(); fig.savefig(Path(out) / f"fig_recovery_{model}.pdf"); plt.close(fig)


def fig_post3d(res, model, out, case=1, ds=1):
    """Joint posterior of (log10 p1, log10 p2, tau) for one dataset, coloured by log10 delta (4-D), truth starred."""
    M = load_all(res, model, case); tv = A.truth_vec(case, 4 if model == "4d" else 3)
    meths = [m for m in ("GPS-ABC", "DNN-ABC-rho", "NPE") if ds in M[m]]
    fig = plt.figure(figsize=(4.7 * len(meths) + 0.8, 4.6))
    for j, m in enumerate(meths):
        ax = fig.add_subplot(1, len(meths), j + 1, projection="3d"); d = M[m][ds][0]; d = d[np.random.default_rng(0).choice(len(d), min(1500, len(d)), replace=False)]
        sc = ax.scatter(d[:, 0], d[:, 1], d[:, 2], c=d[:, 3] if d.shape[1] > 3 else COL[m], s=5, alpha=.55, cmap="viridis", vmin=np.log10(.5), vmax=np.log10(2))
        ax.scatter(*tv[:3], marker="*", s=220, color="gold", edgecolor="k", linewidth=.8, zorder=10)
        ax.set_xlabel(LAB["log10 p1"], labelpad=2); ax.set_ylabel(LAB["log10 p2"], labelpad=2); ax.set_zlabel(LAB["tau"], labelpad=1); ax.set_title(m, fontsize=10)
        ax.set_xlim(-11, -7); ax.set_ylim(-11, -7); ax.set_zlim(0.1, 19.9); ax.view_init(18, -62); ax.tick_params(labelsize=7, pad=0)
        ax.set_xticks([-11, -9, -7]); ax.set_yticks([-11, -9, -7]); ax.set_zticks([0, 10, 20])
    cb = fig.colorbar(sc, ax=fig.axes, shrink=.55, pad=.08); cb.set_label(LAB["log10 delta"])
    fig.savefig(Path(out) / f"fig_post3d_{model}.pdf"); plt.close(fig)


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
    z = np.load(f); D = 3 if model == "3d" else 4; rows = [("DNN-ABC", z["rank_dnnabc"][z["nonzero"]], int(z["L_dnnabc"]))]
    g = Path(res) / f"sbc_{model}_rho.npz"
    if g.exists(): zr = np.load(g); rows.append(("DNN-ABC-rho", zr["rank_dnnabc"][zr["nonzero"]], int(zr["L_dnnabc"])))
    if "rank_npe" in z: rows.append(("NPE", z["rank_npe"], 1000))
    fig, axs = plt.subplots(len(rows), D, figsize=(2.5 * D, 1.9 * len(rows) + .4), sharey=True)
    for r, (nm, R_, L) in enumerate(rows):
        for k in range(D):
            ax = axs[r, k]; h = np.histogram(R_[:, k], bins=np.linspace(-.5, L + .5, 11))[0]
            ax.bar(range(10), h / h.sum(), color=COL[nm]); ax.axhline(.1, color="k", ls="--", lw=.7); ax.set_xticks([])
            lab = nm.replace("-rho", r"-$\rho$"); ax.set_title(f"{lab}: {LAB[A.NAMES[k]]}", fontsize=8)
    fig.tight_layout(); fig.savefig(Path(out) / f"fig_sbc_{model}.pdf"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--results", default="results"); ap.add_argument("--out", default="figures")
    a = ap.parse_args(); Path(a.out).mkdir(exist_ok=True, parents=True)
    fig_recovery(a.results, a.model, a.out)
    if a.model == "3d": fig_post3d(a.results, a.model, a.out)
    else: fig_post3d(a.results, a.model, a.out)      # first three coordinates (p1, p2, tau) of the 4-D posterior
    fig_fidelity(a.results, a.model, a.out); fig_sbc(a.results, a.model, a.out)
    print("figures written to", a.out)
