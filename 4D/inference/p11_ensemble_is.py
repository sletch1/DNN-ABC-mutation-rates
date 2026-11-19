"""Does propagating the surrogate's own parameter uncertainty change calibration? (status item 8.3; Reiser et al. 2025's epistemic term)

The SBC of the paper holds the trained network fixed. Here a bootstrap ensemble of M networks (model/train_ensemble.py: each trained on a resample of
the training designs, different seed) stands for the posterior over the surrogate's parameters, and the surrogate likelihood is averaged over it, in two
ways: E-Lik, the mean of the member likelihoods (the likelihood of a model that integrates over the networks), and E-LogLik, the mean of the member
log-likelihoods (the geometric mean). Each is sampled EXACTLY, by importance sampling from the prior (2e6 uniform draws in the box, truncated-Gaussian prior
as weights), so the sampler cannot be the cause of any miscalibration, and scored by SBC over the same K prior draws and cached data sets as the base
variant (results/p3/data.npz). The deployed single network is scored the same way as the reference. Independent-Gaussian predictive density, hurdle
atom for dbar = 0, as in the base variant.
    python inference/p11_ensemble_is.py --members "results/ensemble/member_*.pt" [--N 2000000] [--K 500]
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "model"))
from sim import settings as S
import dnn_abc
from surrogate import HurdleSurrogate
OUT = HERE.parent / "results" / "p11"; OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]


def tabulate(path, P):
    """Predict (log(1-p0), log p0, mu, 1/sd, log-sd-sum) of one network at all prior draws, in float32."""
    sur = HurdleSurrogate(path); n = len(P)
    l1p = np.empty(n, np.float32); lp0 = np.empty(n, np.float32); mu = np.empty((n, 2), np.float32); isd = np.empty((n, 2), np.float32); lsd = np.empty(n, np.float32)
    for i in range(0, n, 200_000):
        p0, m, sd = sur.predict(P[i:i + 200_000]); p0 = np.clip(p0, 1e-12, 1 - 1e-12)
        l1p[i:i + 200_000], lp0[i:i + 200_000], mu[i:i + 200_000], isd[i:i + 200_000], lsd[i:i + 200_000] = np.log1p(-p0), np.log(p0), m, 1 / sd, np.log(sd).sum(1)
    return dict(l1p=l1p, lp0=lp0, mu=mu, isd=isd, lsd=lsd)


def loglik(T, f):
    if f[0] <= 0: return T["lp0"].astype(np.float64)
    z = np.log10(f).astype(np.float32)
    r = ((z[0] - T["mu"][:, 0]) * T["isd"][:, 0]) ** 2 + ((z[1] - T["mu"][:, 1]) * T["isd"][:, 1]) ** 2
    return (T["l1p"] - np.float32(np.log(2 * np.pi)) - T["lsd"] - np.float32(0.5) * r).astype(np.float64)


def summarise(U, V, nz, box):
    out = {}
    for tag, m in (("all", np.ones(len(U), bool)), ("dbar>0", nz)):
        o = {}
        for j, nm in enumerate(NAMES):
            u = U[m, j]; n = len(u); h = np.histogram(u, bins=np.linspace(0, 1, 11))[0]; e = n / 10
            c95 = float(np.mean((u >= .025) & (u <= .975))); se = np.sqrt(.95 * .05 / n)
            o[nm] = dict(cov95=c95, z=float((c95 - .95) / se), chi2_p=float(1 - chi2.cdf(((h - e) ** 2 / e).sum(), 9)), cov50=float(np.mean((u >= .25) & (u <= .75))),
                         cov90=float(np.mean((u >= .05) & (u <= .95))), contraction=float(1 - V[m, j].mean() / ((box[j, 1] - box[j, 0]) ** 2 / 12)))
        o["n"] = int(m.sum()); out[tag] = o
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--members", default="results/ensemble/member_*.pt"); ap.add_argument("--deployed", default="results/surrogate_4d.pt")
    ap.add_argument("--N", type=int, default=2_000_000); ap.add_argument("--K", type=int, default=500); a = ap.parse_args()
    D = np.load(HERE.parent / "results" / "p3" / "data.npz"); TH, F, MOM = D["theta"][:a.K], D["feat"][:a.K], D["mom"][:a.K]; K = len(TH)
    box = np.array(S.BOX_4D); rng = np.random.default_rng(1); P = box[:, 0] + rng.random((a.N, 4)) * (box[:, 1] - box[:, 0])
    files = sorted(glob.glob(a.members)); print(len(files), "members", flush=True); t0 = time.time()
    Tm = [tabulate(f, P) for f in files]; Td = tabulate(a.deployed, P); print("tabulated", round(time.time() - t0), "s", flush=True)
    variants = {"single (deployed)": None, "E-Lik (ensemble mean likelihood)": None, "E-LogLik (ensemble mean log-likelihood)": None, "ensemble member 0 alone": None}
    U = {v: np.empty((K, 4)) for v in variants}; V = {v: np.empty((K, 4)) for v in variants}; ESS = {v: np.empty(K) for v in variants}
    spread = np.empty(K)
    for k in range(K):
        lp = -0.5 * (((P - dnn_abc.prior_centres(MOM[k], 4)) / 20.0) ** 2).sum(1)
        LL = np.stack([loglik(T, F[k]) for T in Tm]); ld = loglik(Td, F[k])
        lls = {"single (deployed)": ld, "E-Lik (ensemble mean likelihood)": logsumexp(LL, axis=0) - np.log(len(Tm)), "E-LogLik (ensemble mean log-likelihood)": LL.mean(0), "ensemble member 0 alone": LL[0]}
        for v, ll in lls.items():
            lw = ll + lp; lw -= lw.max(); w = np.exp(lw); w /= w.sum(); ESS[v][k] = 1 / (w ** 2).sum(); m = w @ P; V[v][k] = w @ (P - m) ** 2
            U[v][k] = [w[P[:, j] < TH[k, j]].sum() for j in range(4)]
        sd_ = LL.std(0); spread[k] = float(np.mean(sd_[np.isfinite(sd_)]))      # mean across draws of the between-member sd of the log-likelihood
        if k % 50 == 0: print("k", k, round(time.time() - t0), "s", flush=True)
    nz = F[:, 0] > 0
    res = {"N": a.N, "K": K, "members": len(files), "seconds": time.time() - t0, "member_loglik_sd_median": float(np.median(spread)),
           "ess_median": {v: float(np.median(ESS[v])) for v in variants}, "ess_min": {v: float(ESS[v].min()) for v in variants},
           "scores": {v: summarise(U[v], V[v], nz, box) for v in variants}}
    np.savez_compressed(OUT / "ensemble_is.npz", **{("u_" + str(i)): U[v] for i, v in enumerate(variants)}, nonzero=nz)
    json.dump(res, open(OUT / "ensemble_is.json", "w"), indent=1); print(json.dumps({v: res["scores"][v]["all"]["log10 p2"] for v in variants}, indent=1))


if __name__ == "__main__":
    main()
