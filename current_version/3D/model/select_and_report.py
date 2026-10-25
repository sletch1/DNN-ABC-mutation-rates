"""Pick the architecture, train + save the final 3-D surrogate, make figures and a PDF report.
    python model/select_and_report.py --data data/slow_data_3D.csv --search results/arch_search --out results
SELECTION RULE (fixed before looking at test numbers): among architectures, take the best mean
VALIDATION hurdle loss; the chosen model is the cheapest (lowest single-query latency) whose
mean validation loss is within one standard error of that best. Test metrics are reported, never used to select.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_search as A


def agg(df):
    d = df[~df.arch.str.startswith("GP_")]
    cols = ["val_loss", "test_x_floor", "test_rmse", "test_nll", "test_ll0", "test_brier0", "test_auc0",
            "test_cov95_raw", "test_cov95_conf", "test_width95", "params", "train_s", "latency_us", "batch1e5_ms", "epochs"]
    g = d.groupby("arch")[cols]
    m, s = g.mean(), g.std(ddof=1); n = g.size()
    return m, s, n


def choose(m, s, n):
    best = m.val_loss.idxmin(); se = float(np.nan_to_num(s.val_loss[best] / np.sqrt(n[best])))
    ok = m[m.val_loss <= m.val_loss[best] + se]
    return best, ok.latency_us.idxmin(), float(se)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--search", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out)
    raw = pd.read_csv(a.search + "_raw.csv"); meta = json.load(open(a.search + "_meta.json"))
    m, s, n = agg(raw); best, chosen, se = choose(m, s, n)
    gp = raw[raw.arch.str.startswith("GP_")].iloc[0]

    # ---- final model: chosen architecture, seed 0, saved with scalers + conformal scale
    X, zero, y, des, idx = A.load(a.data); floor = A.noise_floor(y, zero, des, idx)
    net, sc, secs, ep, _ = A.fit(A.ARCHS[chosen], X, zero, y, idx, seed=0)
    ev = A.evaluate(net, sc, X, zero, y, des, idx, floor)
    D = X.shape[1]; tag = f"{D}d"
    box = [[-11, -7], [-11, -7], [0.1, 19.9]] + ([[float(np.log10(.5)), float(np.log10(2))]] if D == 4 else [])
    torch.save(dict(state=net.state_dict(), arch=A.ARCHS[chosen], arch_name=chosen, scalers=sc, in_dim=D,
                    conformal_scale=[ev["conformal_scale_f1"], ev["conformal_scale_f2"]], box=box), out / f"surrogate_{tag}.pt")
    json.dump(dict(arch=chosen, best_val_arch=best, metrics=ev, floor=floor), open(out / f"surrogate_{tag}_metrics.json", "w"), indent=1)

    ii = idx["test"]; p0, mu, sd = A.predict(net, sc, X[ii]); sd = sd * np.array([ev["conformal_scale_f1"], ev["conformal_scale_f2"]])
    nz = zero[ii] < .5

    with PdfPages(out / f"arch_search_report_{tag}.pdf") as pdf:
        # page 1: summary text
        t = f"""{tag.upper()} DNN SURROGATE - ARCHITECTURE SEARCH REPORT

GOAL: replace the GP surrogate in ABC-MCMC. For theta=(log10 p1, log10 p2, tau) predict the
joint distribution of the two ABC features of the J=100 cultures: d_bar (mean of (X/Z)^(1/4)) and s (its sd),
fast and with an honest spread.   Data: {meta['n_rows']:,} rows (10,000 LHS designs x 5 reps), paper box, tp=20, {'delta free' if D == 4 else 'delta=1'}.
{meta['frac_zero']*100:.1f}% of rows have d_bar = 0, so the model is a hurdle model:
P(d_bar=0) (logistic head) + Gaussians on (log10 d_bar, log10 s) for nonzero (mean, log-variance heads).
Split BY DESIGN point 80/10/10 (unseen theta), {len(raw)-1} fits = {len(m)} architectures x 5 seeds, {meta['cores']} cores, {meta['wall_s']/60:.1f} min.
Noise floor (irreducible within-design variance), test: log10 d_bar {floor['test'][0]:.2e}, log10 s {floor['test'][1]:.2e}.

SELECTION (rule fixed in advance): best mean validation loss = {best};
chosen = cheapest model within 1 SE ({se:.4f}) of it = {chosen}
Chosen model (seed 0, saved to surrogate_3d.pt) on held-out TEST designs:
  nonzero part (d_bar | s): MSE = {ev['test_f1_x_floor']:.2f} | {ev['test_f2_x_floor']:.2f} x noise floor, RMSE {ev['test_f1_rmse']:.4f} | {ev['test_f2_rmse']:.4f} (log10),
                95% coverage {ev['test_f1_cov95_conf']:.3f} | {ev['test_f2_cov95_conf']:.3f} after conformal (raw {ev['test_f1_cov95_raw']:.3f} | {ev['test_f2_cov95_raw']:.3f})
  zero part   : log-loss {ev['test_ll0']:.4f}, Brier {ev['test_brier0']:.4f}, AUC {ev['test_auc0']:.4f}
GP baseline (GPS-ABC style, 300 pts, nonzero part only, cannot model the zero atom):
  log10 d_bar only: RMSE {gp.test_rmse:.4f} = {gp.test_x_floor:.2f} x floor, 95% coverage {gp.test_cov95_raw:.3f}, fit {gp.train_s:.0f}s, query {gp.latency_us:.0f} us
"""
        f = plt.figure(figsize=(8.5, 11)); f.text(.05, .97, t, va="top", family="monospace", fontsize=7.4); pdf.savefig(f); plt.close(f)
        # page 2: table
        T = m[["params", "val_loss", "test_x_floor", "test_rmse", "test_nll", "test_ll0", "test_auc0", "test_cov95_conf", "latency_us", "batch1e5_ms", "train_s"]].copy()
        T["x_floor_sd"] = s.test_x_floor; T = T.sort_values("val_loss"); T.insert(0, "arch", T.index)
        f, ax = plt.subplots(figsize=(11, 6)); ax.axis("off")
        cell = [[r.arch] + [f"{v:,.0f}" if k == "params" else f"{v:.4g}" for k, v in r.drop("arch").items()] for _, r in T.iterrows()]
        tb = ax.table(cellText=cell, colLabels=["arch"] + list(T.columns[1:]), loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(5.5); tb.scale(1, 1.4)
        ax.set_title("Mean over 5 seeds (sorted by validation loss; test_* are held-out designs; latency = batch 1, microseconds)", fontsize=8)
        pdf.savefig(f); plt.close(f)
        # page 3: tradeoff plots
        f, axs = plt.subplots(2, 2, figsize=(11, 8.5))
        for ax, (xk, xl) in zip(axs.ravel(), [("latency_us", "latency, batch 1 (us)"), ("params", "parameters"), ("train_s", "train time (s)"), ("batch1e5_ms", "100k-point batch (ms)")]):
            ax.errorbar(m[xk], m.test_x_floor, yerr=s.test_x_floor, fmt="o", ms=4)
            for k in m.index: ax.annotate(k.replace("_", " "), (m[xk][k], m.test_x_floor[k]), fontsize=5)
            ax.scatter(m[xk][chosen], m.test_x_floor[chosen], s=90, facecolors="none", edgecolors="r", label="chosen"); ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_ylabel("test MSE / noise floor"); ax.legend(fontsize=6)
        f.suptitle("Accuracy vs cost (1.0 = irreducible noise)"); f.tight_layout(); pdf.savefig(f); plt.close(f)
        # page 4: fit and calibration, both features + zero head
        f, axs = plt.subplots(2, 3, figsize=(12, 7.5))
        for k, nm in enumerate(["log10 d_bar", "log10 s"]):
            ax = axs[0, k]; ax.scatter(y[ii][nz, k], mu[nz, k], s=2, alpha=.3); lim = [y[ii][nz, k].min(), y[ii][nz, k].max()]; ax.plot(lim, lim, "r-", lw=.8)
            ax.set_xlabel(f"true {nm}"); ax.set_ylabel("predicted mean"); ax.set_title(f"{nm}: parity (unseen designs)")
            ax = axs[1, k]; zs = (y[ii][nz, k] - mu[nz, k]) / sd[nz, k]; ax.hist(zs, bins=80, density=True); g = np.linspace(-4, 4, 200)
            ax.plot(g, np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), "r"); ax.set_xlim(-5, 5); ax.set_title(f"{nm}: standardised residuals vs N(0,1)")
        ax = axs[0, 2]; bins = np.quantile(p0, np.linspace(0, 1, 11)); b = np.clip(np.digitize(p0, bins[1:-1]), 0, 9)
        ax.plot([p0[b == k].mean() for k in range(10)], [zero[ii][b == k].mean() for k in range(10)], "o-"); ax.plot([0, 1], [0, 1], "r--")
        ax.set_xlabel("predicted P(d_bar=0)"); ax.set_ylabel("observed frequency"); ax.set_title("zero head reliability")
        ax = axs[1, 2]; err = np.abs(y[ii][nz, 0] - mu[nz, 0]); ax.scatter(sd[nz, 0], err, s=2, alpha=.3); ax.plot([0, sd[nz, 0].max()], [0, 1.96 * sd[nz, 0].max()], "r--", lw=.8)
        ax.set_xlabel("predicted sd (d_bar)"); ax.set_ylabel("|error|"); ax.set_title("predicted sd tracks error")
        f.tight_layout(); pdf.savefig(f); plt.close(f)
        # page 5: 3D surface of predicted log10 d_bar and P0
        g = np.linspace(0, 1, 18); P1, P2, TT = np.meshgrid(g, g, g, indexing="ij")
        G = np.column_stack([-11 + 4 * P1.ravel(), -11 + 4 * P2.ravel(), 0.1 + 19.8 * TT.ravel()] + ([np.zeros(P1.size)] if D == 4 else [])).astype(np.float32)
        gp0, gmu, gsd = A.predict(net, sc, G); gmu = gmu[:, 0]
        f = plt.figure(figsize=(11, 5.5))
        for k, (val, ttl, cm) in enumerate([(gmu, "predicted mean log10 d_bar (nonzero)" + (", delta=1 slice" if D == 4 else ""), "viridis"), (gp0, "predicted P(d_bar = 0)", "magma")]):
            ax = f.add_subplot(1, 2, k + 1, projection="3d"); sc_ = ax.scatter(G[:, 0], G[:, 1], G[:, 2], c=val, s=6, cmap=cm, alpha=.5)
            f.colorbar(sc_, ax=ax, shrink=.55); ax.set_xlabel("log10 p1"); ax.set_ylabel("log10 p2"); ax.set_zlabel("tau"); ax.set_title(ttl, fontsize=9); ax.view_init(22, -55)
        pdf.savefig(f); plt.close(f)
    print("chosen", chosen, "best", best)


if __name__ == "__main__":
    main()
