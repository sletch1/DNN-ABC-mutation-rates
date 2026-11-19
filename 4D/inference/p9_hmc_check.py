"""Gradient-based sampler on the surrogate posterior (status item 3.1).

The random-walk chain of DNN-ABC is overconfident for log10 p2 over prior draws while exact importance resampling of the SAME surrogate
posterior is calibrated (p3_is_check.py). The network is differentiable, so Hamiltonian Monte Carlo can target the same posterior.
This script runs HMC on the cached simulator data of p3_overconfidence.py (results/p3/data.npz, same K prior draws and data sets as
the base SBC) and scores it the same way, so the three samplers (random walk, importance resampling, HMC) are compared on one set of
data sets. Posterior, prior and likelihood are identical to the 'base' variant of p3_overconfidence.py: independent Gaussian predictive
density, conformal-scaled sd, hurdle atom when the data set has no mutants, truncated-Gaussian prior (sd 20) inside the paper's box.

Sampling is done in an unconstrained space u (theta = lo + (hi - lo) * sigmoid(u), with the log-Jacobian added), by HMC with dual-averaging
step-size adaptation and a diagonal mass matrix estimated halfway through warm-up. Two chains per data set (one from the random-walk start,
one from the centre of the box). Reports SBC coverage and rank-uniformity (same `score` as the base variant), contraction, acceptance,
divergences and an effective-sample-size summary.
    python inference/p9_hmc_check.py --surrogate results/surrogate_4d.pt [--K 500] [--workers 12]
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "model"))
from sim import settings as S
import dnn_abc, sbc as SBC
from surrogate import HurdleSurrogate
from p3_overconfidence import score
OUT = HERE.parent / "results" / "p9"; OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]
G = {}


def _init(path, n_iter, warm, thin):
    torch.set_num_threads(1)
    sur = HurdleSurrogate(path); sur.net.double()
    box = np.array(S.BOX_4D)
    G.update(sur=sur, n=n_iter, warm=warm, thin=thin, lo=torch.tensor(box[:, 0]), hi=torch.tensor(box[:, 1]),
             xm=torch.tensor(sur.sc["xm"], dtype=torch.float64), xs=torch.tensor(sur.sc["xs"], dtype=torch.float64),
             ym=torch.tensor(sur.sc["ym"], dtype=torch.float64), ys=torch.tensor(sur.sc["ys"], dtype=torch.float64),
             tc=torch.tensor(sur.tconf, dtype=torch.float64))


def logpost(u, f, mu0):
    """log posterior density in u-space (includes the log-Jacobian of the box transform)."""
    lo, hi = G["lo"], G["hi"]
    th = lo + (hi - lo) * torch.sigmoid(u)
    out = G["sur"].net(((th - G["xm"]) / G["xs"])[None])
    lg, mu, lv = out[0][0], out[1][0], out[2][0]
    p0 = torch.sigmoid(lg).clamp(1e-12, 1 - 1e-12)
    if f[0] <= 0:
        ll = torch.log(p0)
    else:
        m = mu * G["ys"] + G["ym"]; sd = torch.exp(0.5 * lv) * G["ys"] * G["tc"]
        z = torch.log10(torch.tensor(np.maximum(f, 1e-300), dtype=torch.float64))
        ll = torch.log1p(-p0) - np.log(2 * np.pi) - torch.log(sd).sum() - 0.5 * (((z - m) / sd) ** 2).sum()
    lp = -0.5 * (((th - torch.tensor(mu0)) / 20.0) ** 2).sum()
    lj = (torch.log(hi - lo) + torch.nn.functional.logsigmoid(u) + torch.nn.functional.logsigmoid(-u)).sum()
    return ll + lp + lj


def vg(u, f, mu0):
    u = u.detach().requires_grad_(True); v = logpost(u, f, mu0); g, = torch.autograd.grad(v, u)
    return v.detach(), g.detach()


def hmc(u0, f, mu0, rng, n, warm):
    """HMC with dual averaging (target accept 0.9) and a diagonal mass matrix set at the middle of warm-up. Returns u draws, accept rate, divergences."""
    D = len(u0); u = u0.clone(); inv_m = torch.ones(D, dtype=torch.float64)
    v, g = vg(u, f, mu0)
    eps, mu_da, hbar, lbar, t0, gam, kap, delta = 0.1, np.log(10 * 0.1), 0.0, 0.0, 10, 0.05, 0.75, 0.9
    out = np.empty((n, D)); acc = []; div = 0; m_da = 0
    def step(u, v, g, eps, L):
        p = torch.randn(D, dtype=torch.float64) / torch.sqrt(inv_m); h0 = -v + 0.5 * (p ** 2 * inv_m).sum()
        un, pn, gn = u.clone(), p.clone(), g.clone(); vn = v
        for _ in range(L):
            pn = pn + 0.5 * eps * gn; un = un + eps * inv_m * pn
            vn, gn = vg(un, f, mu0); pn = pn + 0.5 * eps * gn
        h1 = -vn + 0.5 * (pn ** 2 * inv_m).sum()
        dh = float(h1 - h0)
        return un, vn, gn, (0.0 if not np.isfinite(dh) else min(1.0, float(np.exp(-dh)))), (not np.isfinite(dh)) or dh > 1000
    for i in range(n):
        L = int(np.clip(round(1.5 / eps * (0.8 + 0.4 * rng.random())), 3, 64))
        un, vn, gn, a, dv = step(u, v, g, eps, L)
        div += int(dv)
        if rng.random() < a and not dv: u, v, g = un, vn, gn
        if i < warm:
            m_da += 1; w = 1.0 / (m_da + t0); hbar = (1 - w) * hbar + w * (delta - a)
            leps = mu_da - np.sqrt(m_da) / gam * hbar; eta = m_da ** (-kap); lbar = eta * leps + (1 - eta) * lbar; eps = float(np.exp(leps))
            if i == warm // 2 - 1:
                var = torch.tensor(np.var(out[max(0, i - 150):i + 1], axis=0) + 1e-6, dtype=torch.float64)
                inv_m = var; eps = 0.1; mu_da = np.log(10 * eps); hbar = lbar = 0.0; m_da = 0
            if i == warm - 1: eps = float(np.exp(lbar))
        else: acc.append(a)
        out[i] = u.numpy()
    return out[warm:], float(np.mean(acc)), div


def _job(args):
    k, f, mom, seed = args
    rng = np.random.default_rng(seed + k); torch.manual_seed(seed + k)
    lo, hi = G["lo"].numpy(), G["hi"].numpy(); mu0 = dnn_abc.prior_centres(mom, 4)
    starts = [dnn_abc.START[:4], 0.5 * (lo + hi)]
    draws, accs, divs = [], [], 0
    for s0 in starts:
        x = np.clip((s0 - lo) / (hi - lo), 1e-3, 1 - 1e-3); u0 = torch.tensor(np.log(x / (1 - x)))
        d, a, dv = hmc(u0, f, mu0, rng, G["n"], G["warm"])
        draws.append(lo + (hi - lo) / (1 + np.exp(-d))); accs.append(a); divs += dv
    th = np.concatenate([d[::G["thin"]] for d in draws])
    # split-chain agreement for log10 p2: |difference of the two chains' medians| / pooled sd
    gap = float(abs(np.median(draws[0][:, 1]) - np.median(draws[1][:, 1])) / max(np.std(np.concatenate([d[:, 1] for d in draws])), 1e-9))
    return th, float(np.mean(accs)), divs, gap


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--surrogate", default="results/surrogate_4d.pt"); ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--iters", type=int, default=2000); ap.add_argument("--warm", type=int, default=500); ap.add_argument("--thin", type=int, default=15)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--seed", type=int, default=2026); a = ap.parse_args()
    D = np.load(HERE.parent / "results" / "p3" / "data.npz"); K = min(a.K, len(D["theta"])); TH, F, MOM = D["theta"][:K], D["feat"][:K], D["mom"][:K]
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.surrogate, a.iters, a.warm, a.thin)) as p:
        res = p.map(_job, [(k, F[k], MOM[k], a.seed) for k in range(K)], chunksize=1)
    R = np.array([SBC.ranks(TH[k], res[k][0]) for k in range(K)]); L = len(res[0][0]); nz = F[:, 0] > 0
    box = np.array(S.BOX_4D); V = np.array([r[0].var(0) for r in res])[nz]
    sc = {"all": score(TH, R, L, label="hmc"), "dbar>0": score(TH, R, L, nz), "L": L, "K": K, "seconds": time.time() - t0,
          "accept_median": float(np.median([r[1] for r in res])), "divergent_transitions_total": int(sum(r[2] for r in res)),
          "data_sets_with_divergences": int(sum(r[2] > 0 for r in res)), "chain_gap_median": float(np.median([r[3] for r in res])),
          "chain_gap_over_0.5": int(sum(r[3] > 0.5 for r in res)),
          "contraction": {n: float(1 - V[:, i].mean() / ((box[i, 1] - box[i, 0]) ** 2 / 12)) for i, n in enumerate(NAMES)}}
    np.savez_compressed(OUT / "hmc_ranks.npz", ranks=R, L=L, theta=TH, nonzero=nz)
    json.dump(sc, open(OUT / "hmc_check.json", "w"), indent=1); print(json.dumps(sc, indent=1))


if __name__ == "__main__":
    main()
