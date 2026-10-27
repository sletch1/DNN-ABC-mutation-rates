"""DNN-ABC: the GPS-ABC Metropolis-Hastings sampler (Lu, Zhu & Wu 2023, get_theta2a.m / ABC_mu2a.m) with the GP
surrogate replaced by the trained hurdle network. Everything that is not the surrogate follows the paper:
priors TN(log10 p_MOM,20,[-11,-7]) x2, TN(10,20,[0.1,19.9]) and (4-D) TN(0,20,[log10 .5, log10 2]);
Gaussian random-walk proposal with covariance adapted during burn-in (see run_chain); 20,000 retained draws after 15,000 discarded iterations (10,000 adaptation + the paper's 5,000 burn-in); start (p1,p2,tau,delta) = (1e-9, 1e-8, 14, 0.8).
There is no refinement step: the network is trained once and the acceptance decision uses its predictive
density directly (no Monte-Carlo draws from the surrogate, so no "unconditional error" threshold xi).

    python inference/dnn_abc.py --surrogate results/surrogate_3d.pt --obs data/obs_3d_case1.npz --out results/dnnabc_3d_case1.npz
"""
import argparse, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri, log_ndtr
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from surrogate import HurdleSurrogate

STEPS = np.array([3.5, 3.5, 4.0, 0.8]); START = np.array([-9.0, -8.0, 14.0, np.log10(0.8)])


def tn_draw(mu, s, lo, hi, u):
    a, b = ndtr((lo - mu) / s), ndtr((hi - mu) / s)
    return float(np.clip(mu + s * ndtri(np.clip(a + u * (b - a), 1e-300, 1 - 1e-16)), lo, hi))


def tn_logZ(mu, s, lo, hi):
    return np.log(max(ndtr((hi - mu) / s) - ndtr((lo - mu) / s), 1e-300))


def prior_centres(mom, D):
    lm = np.log10(max(mom, 1e-11)); return np.array([lm, lm, 10.0, 0.0])[:D]


def run_chain(sur, obs_x, mom, n_iter, rng, sigma=20.0, step_scale=0.05, adapt_until=10000):
    """Random-walk Metropolis--Hastings with the paper's priors and likelihood from the surrogate.
    Proposal: Gaussian random walk whose covariance starts at diag((step_scale x paper widths (3.5,3.5,4,0.8))^2) and, during the
    first `adapt_until` iterations (the burn-in that is discarded), is re-estimated every 250 draws as 2.38^2/D x the sample
    covariance of the last 2,000 draws (adaptive Metropolis, Haario et al. 2001), then FROZEN, so the retained draws come from a
    fixed-proposal chain. Proposals outside the box are rejected (the prior is zero there; the kernel stays symmetric).
    The target is unchanged; only the proposal differs from the paper's component-wise truncated normal, because a diagonal
    kernel mixes poorly along the p2--(p1, tau) ridge. Returns (draws, post-burn-in acceptance rate)."""
    D, box = sur.dim, sur.box
    obs_feat = np.array([np.mean(obs_x), np.std(obs_x, ddof=1)])
    mu0 = prior_centres(mom, D); lo, hi = box[:, 0], box[:, 1]; width = hi - lo
    cov = np.diag((STEPS[:D] * step_scale) ** 2); L = np.linalg.cholesky(cov); cur = START[:D].copy()
    lp = lambda t: float(np.sum(-0.5 * ((t - mu0) / sigma) ** 2))
    lcur, pcur = sur.loglik(cur, obs_feat), lp(cur)
    out = np.empty((n_iter, D), np.float32); acc = 0
    Z = rng.standard_normal((n_iter, D)); U = np.log(rng.random(n_iter))
    for i in range(n_iter):
        if 500 <= i < adapt_until and i % 250 == 0:
            C = np.cov(out[max(0, i - 2000):i].astype(float).T) * 2.38 ** 2 / D + np.diag((0.005 * width) ** 2)
            try: L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError: pass
        can = cur + L @ Z[i]
        if np.all(can >= lo) and np.all(can <= hi):
            lc, pc = sur.loglik(can, obs_feat), lp(can)
            if U[i] < lc - lcur + pc - pcur:
                cur, lcur, pcur = can, lc, pc
                if i >= adapt_until: acc += 1
        out[i] = cur
    return out, acc / max(1, n_iter - adapt_until)


G = {}
def _init(path, obs, n_iter, scale):
    import torch; torch.set_num_threads(1)
    G["sur"] = HurdleSurrogate(path); G["obs"] = dict(np.load(obs)); G["n"] = n_iter; G["scale"] = scale


def _job(i):
    t0 = time.time(); rng = np.random.default_rng(7919 * (i + 1))
    s, a = run_chain(G["sur"], G["obs"]["obs"][i], float(G["obs"]["mom"][i]), G["n"], rng, step_scale=G["scale"])
    return s, a, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", required=True); ap.add_argument("--obs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=35000, help="total: 10,000 adaptation + 5,000 burn-in (both discarded) + 20,000 retained"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--step-scale", type=float, default=0.05, help="initial multiplier on the paper step widths (3.5,3.5,4,0.8); adapted during burn-in")
    a = ap.parse_args()
    with Pool(a.cores, initializer=_init, initargs=(a.surrogate, a.obs, a.iters, a.step_scale)) as p:
        r = p.map(_job, range(a.n), chunksize=1)
    np.savez_compressed(a.out, samples=np.array([x[0] for x in r]), acc=np.array([x[1] for x in r]), secs=np.array([x[2] for x in r]))
    print("mean acc", np.mean([x[1] for x in r]), "mean secs/chain", np.mean([x[2] for x in r]))


if __name__ == "__main__":
    main()
