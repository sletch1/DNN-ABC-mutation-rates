"""Simulation-based calibration (Talts et al. 2018) of DNN-ABC and NPE over PRIOR draws (not fixed truths).
For k = 1..K: draw theta_k ~ Uniform(paper box), simulate one dataset (J=100 cultures, fast simulator, Z0=1, a=1, tp=20),
run the method, and record the rank of each true parameter among L thinned post-burn-in draws. If the posterior is calibrated
the ranks are uniform on {0..L}. Reports a chi-square p-value on 10 rank bins and coverage of central 50/90/95% intervals.
GPS-ABC is not run here: one replicate costs ~1 h of MATLAB, so a few hundred prior draws is out of reach (stated in the paper).
    python inference/sbc.py --model 3d --surrogate results/surrogate_3d.pt --npe results/npe_3d.pkl --K 500 --out results/sbc_3d.npz
"""
import argparse, sys, pickle
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import mbp, settings as S
import dnn_abc
from surrogate import HurdleSurrogate

G = {}
def _init(path, D, n_iter, scale):
    import torch; torch.set_num_threads(1)
    G["sur"], G["D"], G["n"], G["scale"] = HurdleSurrogate(path), D, n_iter, scale


def dataset(rng, th, D, J=100, tp=20.0):
    Z = np.empty(J); X = np.empty(J)
    for j in range(J):
        while True:
            if D == 3: z, x = mbp.mut2stage_rev(rng, 1, 1.0, 1.0, 10 ** th[0], 10 ** th[1], th[2], tp)
            else: z, x = mbp.mut2stage_rev(rng, 1, 1.0, 10 ** th[3], 10 ** th[0], 10 ** th[1], th[2], tp)
            if z > 0: break
        Z[j], X[j] = z, x
    mom = 0.5 * (1 - np.log(max(Z.mean() - X.mean(), 1.0)) / np.log(Z.mean()))
    return (X / Z) ** 0.25, mom


def _job(k):
    D = G["D"]; rng = np.random.default_rng(31337 + k)
    box = np.array(S.BOX_3D if D == 3 else S.BOX_4D); th = box[:, 0] + rng.random(D) * (box[:, 1] - box[:, 0])
    obs, mom = dataset(rng, th, D)
    ch, acc = dnn_abc.run_chain(G["sur"], obs, mom, G["n"], rng, step_scale=G["scale"])
    return th, obs, mom, ch[5000::15], acc            # burn-in 5,000 then thin to ~1,000 draws


def ranks(th, draws):
    return np.array([(draws[:, i] < th[i]).sum() for i in range(len(th))])


def summarise(R, L, names):
    out = {}
    for i, nm in enumerate(names):
        h = np.histogram(R[:, i], bins=np.linspace(-.5, L + .5, 11))[0]; e = len(R) / 10
        out[nm] = dict(chi2_p=float(1 - chi2.cdf(((h - e) ** 2 / e).sum(), 9)),
                       **{f"cov{int(c*100)}": float(np.mean((R[:, i] >= L * (1 - c) / 2) & (R[:, i] <= L * (1 + c) / 2))) for c in (.5, .9, .95)})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=["3d", "4d"], required=True); ap.add_argument("--surrogate", required=True)
    ap.add_argument("--npe", default=None); ap.add_argument("--K", type=int, default=500); ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--cores", type=int, default=None); ap.add_argument("--step-scale", type=float, default=1.0); ap.add_argument("--out", required=True)
    a = ap.parse_args(); D = 3 if a.model == "3d" else 4
    with Pool(a.cores, initializer=_init, initargs=(a.surrogate, D, a.iters, a.step_scale)) as p:
        res = p.map(_job, range(a.K), chunksize=2)
    TH = np.array([r[0] for r in res]); names = ["log10 p1", "log10 p2", "tau", "log10 delta"][:D]
    RD = np.array([ranks(r[0], r[3]) for r in res]); LD = len(res[0][3])
    save = dict(theta=TH, rank_dnnabc=RD, L_dnnabc=LD, acc=np.array([r[4] for r in res]))
    summ = {"DNN-ABC": summarise(RD, LD, names)}
    if a.npe:
        import torch; from npe import sample
        post = pickle.load(open(a.npe, "rb")); RN = []
        for k, r in enumerate(res):
            RN.append(ranks(r[0], sample(post, r[1], 1000, seed=k)))
        RN = np.array(RN); save["rank_npe"] = RN; summ["NPE"] = summarise(RN, 1000, names)
    np.savez_compressed(a.out, **save)
    import json; print(json.dumps(summ, indent=1)); json.dump(summ, open(a.out.replace(".npz", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
