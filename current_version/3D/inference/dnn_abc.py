"""DNN-ABC: the GPS-ABC Metropolis-Hastings sampler (Lu, Zhu & Wu 2023, get_theta2a.m / ABC_mu2a.m) with the GP
surrogate replaced by the trained hurdle network. Everything that is not the surrogate follows the paper:
priors TN(log10 p_MOM,20,[-11,-7]) x2, TN(10,20,[0.1,19.9]) and (4-D) TN(0,20,[log10 .5, log10 2]);
component-wise truncated-normal random-walk proposal with the Hastings correction; 20,000 draws, first 5,000
burn-in; start (p1,p2,tau,delta) = (1e-9, 1e-8, 14, 0.8).
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


def run_chain(sur, obs_x, mom, n_iter, rng, sigma=20.0, step_scale=1.0):
    D, box = sur.dim, sur.box
    obs_feat = np.array([np.mean(obs_x), np.std(obs_x, ddof=1)])
    mu0 = prior_centres(mom, D); steps, cur = STEPS[:D] * step_scale, START[:D].copy()
    lp = lambda t: sum(-0.5 * ((t[k] - mu0[k]) / sigma) ** 2 - tn_logZ(mu0[k], sigma, *box[k]) for k in range(D))
    lcur, pcur = sur.loglik(cur, obs_feat), lp(cur)
    out = np.empty((n_iter, D), np.float32); acc = 0
    U = rng.random((n_iter, D + 1))
    for i in range(n_iter):
        can = np.array([tn_draw(cur[k], steps[k], *box[k], U[i, k]) for k in range(D)])
        lq = sum(tn_logZ(cur[k], steps[k], *box[k]) - tn_logZ(can[k], steps[k], *box[k]) for k in range(D))
        lc, pc = sur.loglik(can, obs_feat), lp(can)
        if np.log(U[i, D]) < lc - lcur + pc - pcur + lq:
            cur, lcur, pcur = can, lc, pc; acc += 1
        out[i] = cur
    return out, acc / n_iter


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
    ap.add_argument("--iters", type=int, default=20000); ap.add_argument("--n", type=int, default=100); ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--step-scale", type=float, default=1.0, help="multiplier on the paper step widths (3.5,3.5,4,0.8); see pilot_steps.py")
    a = ap.parse_args()
    with Pool(a.cores, initializer=_init, initargs=(a.surrogate, a.obs, a.iters, a.step_scale)) as p:
        r = p.map(_job, range(a.n), chunksize=1)
    np.savez_compressed(a.out, samples=np.array([x[0] for x in r]), acc=np.array([x[1] for x in r]), secs=np.array([x[2] for x in r]))
    print("mean acc", np.mean([x[1] for x in r]), "mean secs/chain", np.mean([x[2] for x in r]))


if __name__ == "__main__":
    main()
