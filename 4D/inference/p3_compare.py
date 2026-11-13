"""Per-data-set comparison of the MCMC chain and importance sampling for the self-consistency SBC (first K0 data sets).
Reports, per data set and parameter, chain mean/sd vs IS mean/sd and the two ranks of the truth."""
import sys, json
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sim import settings as S
from surrogate import HurdleSurrogate
import dnn_abc
import p3_overconfidence as P3
from multiprocessing import Pool
K0 = int(sys.argv[1]) if len(sys.argv) > 1 else 48
def work(k):
    torch.set_num_threads(1)
    sur = P3.HurdleSurrogate("results/surrogate_4d.pt"); D = np.load(HERE.parent / "results/p3/data.npz"); TH, MOM = D["theta"], D["mom"]
    g = np.random.default_rng(99); F = None
    for j in range(k + 1):
        p0, mu, sd = sur.predict(TH[j]); f = [0.0, 0.0] if g.random() < p0[0] else 10 ** (mu[0] + sd[0] * g.standard_normal(2))
    F = np.array(f)
    v = P3.Variant(sur, None, 0.0); rng = np.random.default_rng(31337 + k)
    ch, acc = dnn_abc.run_chain(v, P3.fake_obs(*F) if F[0] > 0 else np.array([0.0, 0.0]), MOM[k], 35000, rng, step_scale=0.05)
    c = ch[15000:].astype(float)
    box = np.array(S.BOX_4D); r = np.random.default_rng(1 + k); N = 400000; Pp = box[:, 0] + r.random((N, 4)) * (box[:, 1] - box[:, 0])
    p0, mu, sd = sur.predict(Pp); p0 = np.clip(p0, 1e-12, 1 - 1e-12)
    if F[0] <= 0: ll = np.log(p0)
    else: z = np.log10(F); ll = np.log1p(-p0) - np.log(2 * np.pi) - np.log(sd).sum(1) - 0.5 * (((z - mu) / sd) ** 2).sum(1)
    lp = -0.5 * (((Pp - dnn_abc.prior_centres(MOM[k], 4)) / 20.0) ** 2).sum(1); w = np.exp(ll + lp - (ll + lp).max()); w /= w.sum()
    m = w @ Pp; s = np.sqrt(w @ (Pp - m) ** 2)
    return dict(k=k, zero=bool(F[0] <= 0), truth=TH[k].tolist(), chain_mean=c.mean(0).tolist(), chain_sd=c.std(0).tolist(), is_mean=m.tolist(), is_sd=s.tolist(),
                ess=float(1 / (w ** 2).sum()), acc=float(acc), chain_rank=[float((c[:, j] < TH[k][j]).mean()) for j in range(4)],
                is_rank=[float(w[Pp[:, j] < TH[k][j]].sum()) for j in range(4)])
if __name__ == "__main__":
    with Pool(10) as p: res = p.map(work, range(K0), chunksize=1)
    json.dump(res, open(HERE.parent / "results/p3/compare.json", "w"), indent=1)
    for j, n in enumerate(P3.NAMES):
        cs = np.array([r["chain_sd"][j] / r["is_sd"][j] for r in res if r["ess"] > 300]); dm = np.array([(r["chain_mean"][j] - r["is_mean"][j]) / r["is_sd"][j] for r in res if r["ess"] > 300])
        print(n, "median sd ratio chain/IS", np.median(cs), "mean |dmean|/sd", np.abs(dm).mean())
