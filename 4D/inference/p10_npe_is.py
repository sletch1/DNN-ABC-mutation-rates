"""Importance-corrected NPE with the surrogate likelihood (status item 8.2).

Use the trained NPE as an importance proposal for the surrogate posterior. For each data set with dbar > 0 draw N values theta_j ~ q(theta | x)
from the NPE, weight them by  w_j  proportional to  L_surrogate(x | theta_j) * prior(theta_j) / q(theta_j | x)  and score the weighted posterior over
the SAME K prior draws and cached data sets as the base SBC of p3_overconfidence.py (results/p3/data.npz). The surrogate likelihood, the prior
(truncated Gaussian, sd 20, centred on the moment estimate, inside the paper's box) and the box are those of the 'base' variant, so the target is
exactly the surrogate posterior that the random-walk chain and the exact importance resampler (p3_is_check.py --simulator) also target. The NPE was
trained under a uniform prior on the box, so q(theta | x) already contains that prior; dividing by it leaves surrogate likelihood x paper prior.

Reports, per parameter, the same SBC summaries as the base variant (coverage of the central 50/90/95% intervals from the weighted probability-integral
transform, rank chi-square on ten bins), contraction, and the effective sample size of the weights (a measure of whether the NPE is a usable proposal).
The plain NPE draws on the same data sets are scored alongside, so the effect of the correction is visible. Needs sbi (the NPE is an sbi pickle).
    python inference/p10_npe_is.py [--N 10000] [--K 500] [--workers 4]
"""
import argparse, json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
from scipy.stats import chi2
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "model"))
from sim import settings as S
import dnn_abc
from surrogate import HurdleSurrogate
OUT = HERE.parent / "results" / "p10"; OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]
G = {}


def _init(npe_path, sur_path, N):
    torch.set_num_threads(1)
    G.update(post=pickle.load(open(npe_path, "rb")), sur=HurdleSurrogate(sur_path), N=N)


def _job(args):
    k, f, mom, th = args
    post, sur, N = G["post"], G["sur"], G["N"]
    x = torch.tensor([[np.log10(f[0]), np.log10(f[1])]], dtype=torch.float32)
    torch.manual_seed(1000 + k)
    with torch.no_grad():
        P = post.sample((N,), x=x, show_progress_bars=False)
        lq = post.log_prob(P, x=x).numpy().astype(np.float64)
    P = P.numpy().astype(np.float64)
    p0, mu, sd = sur.predict(P); p0 = np.clip(p0, 1e-12, 1 - 1e-12)
    z = np.log10(f)                                     # independent-Gaussian surrogate likelihood (base variant)
    ll = np.log1p(-p0) - np.log(2 * np.pi) - np.log(sd).sum(1) - 0.5 * (((z - mu) / sd) ** 2).sum(1)
    lp = -0.5 * (((P - dnn_abc.prior_centres(mom, 4)) / 20.0) ** 2).sum(1)
    lw = ll + lp - lq; lw -= lw.max(); w = np.exp(lw); w /= w.sum()
    ess = 1.0 / (w ** 2).sum()
    u_is = np.array([w[P[:, j] < th[j]].sum() for j in range(4)])       # weighted PIT of the truth
    u_npe = np.array([(P[:, j] < th[j]).mean() for j in range(4)])      # the plain NPE draws on the same data set
    m = w @ P; var_is = w @ (P - m) ** 2; var_npe = P.var(0)
    return u_is, u_npe, float(ess), var_is, var_npe


def summarise(U, var, box):
    out = {}
    for j, nm in enumerate(NAMES):
        u = U[:, j]; n = len(u); h = np.histogram(u, bins=np.linspace(0, 1, 11))[0]; e = n / 10
        c95 = float(np.mean((u >= .025) & (u <= .975))); se = np.sqrt(.95 * .05 / n)
        out[nm] = dict(cov95=c95, z=float((c95 - .95) / se), chi2_p=float(1 - chi2.cdf(((h - e) ** 2 / e).sum(), 9)),
                       cov50=float(np.mean((u >= .25) & (u <= .75))), cov90=float(np.mean((u >= .05) & (u <= .95))),
                       contraction=float(1 - var[:, j].mean() / ((box[j, 1] - box[j, 0]) ** 2 / 12)))
    out["n"] = int(len(U)); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--surrogate", default="results/surrogate_4d.pt"); ap.add_argument("--npe", default="results/npe_4d_case1.pkl")
    ap.add_argument("--N", type=int, default=10000); ap.add_argument("--K", type=int, default=500); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    D = np.load(HERE.parent / "results" / "p3" / "data.npz"); TH, F, MOM = D["theta"][:a.K], D["feat"][:a.K], D["mom"][:a.K]
    idx = np.where(F[:, 0] > 0)[0]                      # NPE is trained on dbar > 0 only
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.npe, a.surrogate, a.N)) as p:
        res = p.map(_job, [(int(k), F[k], MOM[k], TH[k]) for k in idx], chunksize=1)
    U_is = np.array([r[0] for r in res]); U_npe = np.array([r[1] for r in res]); ess = np.array([r[2] for r in res])
    V_is = np.array([r[3] for r in res]); V_npe = np.array([r[4] for r in res]); box = np.array(S.BOX_4D)
    out = {"N": a.N, "K_dbar_pos": int(len(idx)), "seconds": time.time() - t0,
           "ess_median": float(np.median(ess)), "ess_min": float(ess.min()), "ess_frac_below_100": float(np.mean(ess < 100)), "ess_frac_below_1000": float(np.mean(ess < 1000)),
           "npe_importance_corrected": summarise(U_is, V_is, box), "npe_plain": summarise(U_npe, V_npe, box)}
    np.savez_compressed(OUT / "npe_is.npz", u_is=U_is, u_npe=U_npe, ess=ess, idx=idx)
    json.dump(out, open(OUT / "npe_is.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
