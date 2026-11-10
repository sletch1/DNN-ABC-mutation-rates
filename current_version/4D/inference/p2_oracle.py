"""Problem 2: an independent reference ("oracle") posterior that uses neither a GP nor a neural surrogate.
Synthetic-likelihood ABC-MCMC (Wood 2010; Price et al. 2018) with the fast simulator in the loop: at each proposal theta' we simulate
n_s data sets of J=100 cultures, take the features (log10 dbar, log10 s) of those with dbar>0, fit a bivariate Gaussian, and score the
observed features; the zero-probability factor is (1 - p0_hat), p0_hat = (zeros + .5)/(n_s + 1) (observed dbar>0 only).
Same priors, box and adaptive-covariance random walk as DNN-ABC, so the ONLY difference is the likelihood.
The Gaussian-on-logs synthetic likelihood is itself an approximation (the feature checks show log s is skewed), so this is a
reference, not the truth; the paper says so. One chain per data set, one core each.
    python inference/p2_oracle.py --obs data/obs_4d_case1.npz --case 1 --idx 0 1 2 3 --n_s 100 --iters 12000
"""
import argparse, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sim import mbp, settings as S
import dnn_abc
from dyncores import wait_for_core
G = {}


def sl_loglik(rng, theta, f_obs, n_s, J=100):
    th = np.asarray(theta, float); F = np.empty((n_s, 2)); nz = 0
    for g in range(n_s):
        y = np.array([mbp.draw_stat(rng, th, "4d", 1, 1.0, 20.0, 4, False) for _ in range(J)])
        m, s = y.mean(), y.std(ddof=1)
        if m > 0 and s > 0: F[nz] = np.log10([m, s]); nz += 1
    p0 = (n_s - nz + .5) / (n_s + 1)
    if nz < 10: return -1e10
    mu = F[:nz].mean(0); C = np.cov(F[:nz].T) + 1e-9 * np.eye(2); d = np.log10(f_obs) - mu
    return float(np.log1p(-p0) - 0.5 * np.linalg.slogdet(2 * np.pi * C)[1] - 0.5 * d @ np.linalg.solve(C, d))


def chain(args):
    c, i, obs_x, mom, start, n_s, n_iter, adapt = args
    wait_for_core(jitter=5); t0 = time.time(); rng = np.random.default_rng(555 + 1000 * c + i)
    box = np.array(S.BOX_4D); lo, hi = box[:, 0], box[:, 1]; width = hi - lo; mu0 = dnn_abc.prior_centres(mom, 4)
    f_obs = np.array([obs_x.mean(), obs_x.std(ddof=1)]); lp = lambda t: float(np.sum(-0.5 * ((t - mu0) / 20.0) ** 2))
    cur = np.asarray(start, float); lcur = sl_loglik(rng, cur, f_obs, n_s); pcur = lp(cur)
    L = np.linalg.cholesky(np.diag((dnn_abc.STEPS * 0.05) ** 2)); out = np.empty((n_iter, 4)); acc = 0
    for it in range(n_iter):
        if 500 <= it < adapt and it % 250 == 0:
            C = np.cov(out[max(0, it - 2000):it].T) * 2.38 ** 2 / 4 + np.diag((0.005 * width) ** 2)
            try: L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError: pass
        can = cur + L @ rng.standard_normal(4)
        if np.all(can >= lo) and np.all(can <= hi):
            lc = sl_loglik(rng, can, f_obs, n_s); pc = lp(can)
            if np.log(rng.random()) < lc - lcur + pc - pcur:
                cur, lcur, pcur = can, lc, pc
                if it >= adapt: acc += 1
        out[it] = cur
        if it % 1000 == 0: print(f"case {c} idx {i}: it {it}, {time.time()-t0:.0f}s, acc {acc/max(1,it-adapt):.2f}", flush=True)
    return c, i, out[adapt:], acc / max(1, n_iter - adapt), time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--idx", type=int, nargs="+", default=[0, 1, 2, 3]); ap.add_argument("--cases", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n_s", type=int, default=100); ap.add_argument("--iters", type=int, default=12000); ap.add_argument("--adapt", type=int, default=4000)
    ap.add_argument("--probe", action="store_true", help="timing probe only"); ap.add_argument("--outdir", default=str(HERE.parent / "results" / "oracle")); a = ap.parse_args()
    Path(a.outdir).mkdir(exist_ok=True, parents=True); jobs = []
    for c in a.cases:
        O = dict(np.load(HERE.parent / "data" / f"obs_4d_case{c}.npz")); dn = np.load(HERE.parent / "results" / f"dnnabc_4d_case{c}.npz")["samples"]
        for i in a.idx: jobs.append((c, i, O["obs"][i], float(O["mom"][i]), np.median(dn[i], 0), a.n_s, 300 if a.probe else a.iters, 100 if a.probe else a.adapt))
    with Pool(len(jobs)) as p: res = p.map(chain, jobs, chunksize=1)
    for c, i, s, acc, secs in res:
        np.savez_compressed(Path(a.outdir) / f"oracle_case{c}_{i}.npz", samples=s, acc=acc, secs=secs, n_s=a.n_s)
        print(f"case {c} idx {i}: acc {acc:.3f}, {secs/3600:.2f} h")
