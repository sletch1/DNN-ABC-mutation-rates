"""Problem 12: does the surrogate pipeline hold up better than NPE when the simulation budget is small?
Both methods are trained on the SAME simulator runs: the first n training-pool designs (each with its 5 replicate data sets),
n designs in total. NPE-Z (all rows, zero indicator) uses them with sbi's own 10% internal validation split. The surrogate uses the
first 0.9n designs to train and the last 0.1n for early stopping and conformal calibration, so neither method sees more than n designs.
DNN-ABC is sampled EXACTLY (importance resampling of 1e6 uniform prior draws with the surrogate's correlated hurdle likelihood), so
the chain cannot be the cause of any difference.  Both are scored on the same K held-out test data sets by simulation-based
calibration (rank uniformity, 95% coverage) and posterior contraction.
    python inference/p12_budget.py --data data/slow_data_4D.csv --n 1500 --out results/budget/p12_n1500.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.stats import chi2
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "model"))
import arch_search as A
from surrogate import HurdleSurrogate
from npe import BOX
from p7_extra_features import fit
NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]


def zfeat(y, zero):
    return np.column_stack([zero, np.where(zero[:, None] > .5, 0, y)]).astype(np.float32)


def score(U, V, box, pos):
    """U: (K,4) rank in [0,1] of the truth under each posterior; V: (K,4) posterior variance."""
    out = {}
    for sel, tag in ((np.ones(len(U), bool), "all"), (pos, "dbar_pos")):
        o = {}
        for j, (lo, hi) in enumerate(box):
            u = U[sel, j]; h = np.histogram(u, bins=np.linspace(0, 1, 11))[0]; e = len(u) / 10
            o[NAMES[j]] = dict(cov95=float(np.mean((u >= .025) & (u <= .975))), chi2_p=float(1 - chi2.cdf(((h - e) ** 2 / e).sum(), 9)),
                               contraction=float(1 - V[sel, j].mean() / ((hi - lo) ** 2 / 12)), n=int(sel.sum()))
        out[tag] = o
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=300); ap.add_argument("--N", type=int, default=1_000_000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--arch", default="res_128x3_gelu"); a = ap.parse_args()
    torch.set_num_threads(a.threads); X, zero, y, des, idx = A.load(a.data); floor = A.noise_floor(y, zero, des, idx); box = BOX[4]; res = dict(n=a.n, seed=a.seed)
    pool = np.unique(des[idx["train"]])[:a.n]; ntr = int(round(.9 * a.n)); tr_d, va_d = pool[:ntr], pool[ntr:]
    idx2 = dict(idx); idx2["train"] = idx["train"][np.isin(des[idx["train"]], tr_d)]; idx2["val"] = idx["train"][np.isin(des[idx["train"]], va_d)]
    rows_all = np.concatenate([idx2["train"], idx2["val"]]); res["n_designs"] = int(len(pool)); res["n_rows"] = int(len(rows_all))
    # ---- test sets: one data set per held-out design
    rng = np.random.default_rng(1); te = idx["test"]; first = np.unique(des[te], return_index=True)[1]; te = te[first]; te = rng.choice(te, min(a.K, len(te)), replace=False)
    TH, pos = X[te], zero[te] < .5
    # ---- surrogate: train, then exact importance resampling
    t0 = time.time(); arch = dict(A.ARCHS[a.arch], corr=True); net, sc, secs, ep, _ = A.fit(arch, X, zero, y, idx2, seed=a.seed); ev = A.evaluate(net, sc, X, zero, y, des, idx2, floor)
    res["sur_train_s"] = time.time() - t0; res["sur_test_x_floor"] = [ev["test_f1_x_floor"], ev["test_f2_x_floor"]]
    path = Path(a.out).with_suffix(".sur.pt"); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=net.state_dict(), arch=arch, arch_name=a.arch, scalers=sc, in_dim=4, conformal_scale=[ev["conformal_scale_f1"], ev["conformal_scale_f2"]], box=[list(b) for b in box]), path)
    sur = HurdleSurrogate(str(path)); lo, hi = np.array(box).T; g = np.random.default_rng(5); P = lo + g.random((a.N, 4)) * (hi - lo)
    p0 = np.empty(a.N); mu = np.empty((a.N, 2)); sd = np.empty((a.N, 2)); rho = np.empty(a.N)
    for i in range(0, a.N, 200_000):
        p0[i:i + 200_000], mu[i:i + 200_000], sd[i:i + 200_000] = sur.predict(P[i:i + 200_000]); rho[i:i + 200_000] = np.broadcast_to(sur.rho, p0[i:i + 200_000].shape)
    p0 = np.clip(p0, 1e-12, 1 - 1e-12); lsd = np.log(sd).sum(1); K = len(te); U = np.empty((K, 4)); V = np.empty((K, 4)); ess = []
    for k, i in enumerate(te):
        if zero[i] > .5: ll = np.log(p0)
        else:
            zz = (y[i][None, :] - mu) / sd
            ll = np.log1p(-p0) - np.log(2 * np.pi) - lsd - 0.5 * np.log1p(-rho ** 2) - 0.5 * (zz[:, 0] ** 2 - 2 * rho * zz[:, 0] * zz[:, 1] + zz[:, 1] ** 2) / (1 - rho ** 2)
        w = np.exp(ll - ll.max()); w /= w.sum(); ess.append(1 / (w ** 2).sum()); m = w @ P; V[k] = w @ (P - m) ** 2
        for j in range(4): U[k, j] = w[P[:, j] < TH[k, j]].sum()
    res["dnn_abc_exact"] = score(U, V, box, pos); res["dnn_abc_ess_median"] = float(np.median(ess))
    print("DNN-ABC exact done", json.dumps({k: round(v["cov95"], 3) for k, v in res["dnn_abc_exact"]["all"].items()}), flush=True)
    # ---- NPE-Z on the same runs
    F = zfeat(y, zero); t0 = time.time(); npe = fit(X[rows_all], F[rows_all], box, a.seed); res["npe_train_s"] = time.time() - t0
    L = 1000; U2 = np.empty((K, 4)); V2 = np.empty((K, 4))
    for k, i in enumerate(te):
        s = npe.sample((L,), x=torch.tensor(F[i:i + 1]), show_progress_bars=False).numpy(); U2[k] = (s < TH[k]).mean(0); V2[k] = s.var(0)
    res["npe_z"] = score(U2, V2, box, pos)
    print("NPE-Z done", json.dumps({k: round(v["cov95"], 3) for k, v in res["npe_z"]["all"].items()}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1); print("done", a.out)


if __name__ == "__main__":
    main()
