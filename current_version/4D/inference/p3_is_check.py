"""Independent check of the self-consistency SBC without MCMC: importance sampling from the prior (2e6 uniform draws) with the
surrogate's likelihood as weights. For each of the K prior draws and features drawn from the surrogate's own predictive density
(same seeds as p3_overconfidence.py selfsbc) it computes the weighted PIT of the truth and the 95%-interval coverage.
If this is calibrated, the MCMC sampler is at fault; if it shows the same shortfall, the check itself or the surrogate density is.
    python inference/p3_is_check.py --surrogate results/surrogate_4d.pt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sim import settings as S
from surrogate import HurdleSurrogate
import dnn_abc
NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--surrogate", default="results/surrogate_4d.pt"); ap.add_argument("--N", type=int, default=2_000_000); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--simulator", action="store_true", help="score the simulator-generated features in data.npz with the paper's correlated likelihood (the main SBC), instead of self-generated features"); a = ap.parse_args()
    torch.set_num_threads(a.threads); sur = HurdleSurrogate(a.surrogate); D = np.load(HERE.parent / "results" / "p3" / "data.npz"); TH, MOM = D["theta"], D["mom"]; K = len(TH)
    box = np.array(S.BOX_4D); rng = np.random.default_rng(1); P = box[:, 0] + rng.random((a.N, 4)) * (box[:, 1] - box[:, 0])
    p0 = np.empty(a.N); mu = np.empty((a.N, 2)); sd = np.empty((a.N, 2))
    for i in range(0, a.N, 200_000): p0[i:i + 200_000], mu[i:i + 200_000], sd[i:i + 200_000] = sur.predict(P[i:i + 200_000])
    p0 = np.clip(p0, 1e-12, 1 - 1e-12); lsd = np.log(sd).sum(1)
    g = np.random.default_rng(99); F = np.empty((K, 2))
    if a.simulator: F = D["feat"].copy(); r = float(sur.rho[0])
    for k in (range(0) if a.simulator else range(K)):    # same generation as run_variant(selfc=True)
        p0k, muk, sdk = sur.predict(TH[k])
        F[k] = [0.0, 0.0] if g.random() < p0k else 10 ** (muk[0] + sdk[0] * g.standard_normal(2))
    cov = {n: [] for n in NAMES}; ess = []
    for k in range(K):
        if F[k, 0] <= 0: ll = np.log(p0)
        elif a.simulator: zz = (np.log10(F[k]) - mu) / sd; ll = np.log1p(-p0) - np.log(2 * np.pi) - lsd - 0.5 * np.log1p(-r ** 2) - 0.5 * (zz[:, 0] ** 2 - 2 * r * zz[:, 0] * zz[:, 1] + zz[:, 1] ** 2) / (1 - r ** 2)
        else: z = np.log10(F[k]); ll = np.log1p(-p0) - np.log(2 * np.pi) - lsd - 0.5 * (((z - mu) / sd) ** 2).sum(1)
        lp = -0.5 * (((P - dnn_abc.prior_centres(MOM[k], 4)) / 20.0) ** 2).sum(1); w = np.exp(ll + lp - (ll + lp).max()); w /= w.sum(); ess.append(1 / (w ** 2).sum())
        for j, n in enumerate(NAMES): u = w[P[:, j] < TH[k, j]].sum(); cov[n].append((F[k, 0] > 0, 0.025 <= u <= 0.975))
    out = {"N": a.N, "K": K, "ess_median": float(np.median(ess)), "ess_min": float(np.min(ess))}
    for n in NAMES:
        c = np.array(cov[n]); out[n] = dict(cov95_all=float(c[:, 1].mean()), cov95_dbar_pos=float(c[c[:, 0] == 1, 1].mean()), z=float((c[:, 1].mean() - .95) / np.sqrt(.95 * .05 / K)))
    json.dump(out, open(HERE.parent / "results" / "p3" / ("is_check_simulator.json" if a.simulator else "is_check.json"), "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
