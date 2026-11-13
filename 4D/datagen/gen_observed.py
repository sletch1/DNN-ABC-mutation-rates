"""Observed datasets, mirroring simulation2.m (3d/4d) and simulation1.m (1d):
nmc=100 independent single-culture draws (L=1) per seed, statistic
obs_X=(X/Z)^(1/4), moment estimator MOM=0.5*(1-log(mean Nt - mean Xt)/log(mean Nt))
(2-stage) as in the repo. Truth defaults to Study 2 case 1/2 (a_dt=0.8).

  python datagen/gen_observed.py --model 4d --case 1 --nsim 100 --out data/obs_4d_case1.npz
"""
import argparse, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import mbp, settings as S


def one(args):
    seed, kind, truth, tp, nmc, exact, p = args
    rng = np.random.default_rng(seed)
    Nt = np.empty(nmc); Xt = np.empty(nmc)
    for j in range(nmc):
        while True:   # same Z==0 guard as training generation
            if kind == "1d":
                Z, X = mbp.mut_bMBP(rng, S.Z0, S.A, p, tp) if exact else mbp.mut_bMBP_rev(rng, S.Z0, S.A, 1.0, p, tp)
            elif exact:
                Z, X = mbp.mut2stage_exact_delta(rng, S.A, truth["delta"] if kind == "4d" else 1.0,
                                                 truth["p1"], truth["p2"], truth["tau"], tp)
            else:
                d = truth["delta"] if kind == "4d" else 1.0
                Z, X = mbp.mut2stage_rev(rng, S.Z0, S.A, d, truth["p1"], truth["p2"], truth["tau"], tp)
            if Z > 0: break
        Nt[j], Xt[j] = Z, X
    obs = (Xt / Nt) ** (1.0 / S.ROOT)
    mom = 0.5 * (1 - np.log(Nt.mean() - Xt.mean()) / np.log(Nt.mean()))
    return Nt, Xt, obs, mom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["1d", "3d", "4d"], required=True)
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--nsim", type=int, default=100)
    ap.add_argument("--nmc", type=int, default=100)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    truth = S.SPEC_2["truths"][a.case]
    tp = S.tp_study1(10.0 ** -a.p) if a.model == "1d" else S.TP_2
    with Pool(a.cores) as pool:
        r = pool.map(one, [(s, a.model, truth, tp, a.nmc, a.exact, 10.0 ** -a.p) for s in range(1, a.nsim + 1)])
    np.savez_compressed(a.out, Nt=np.array([x[0] for x in r]), Xt=np.array([x[1] for x in r]),
                        obs=np.array([x[2] for x in r]), mom=np.array([x[3] for x in r]),
                        tp=tp, truth=str(truth), model=a.model)
    print("saved", a.out)


if __name__ == "__main__":
    main()
