"""Run ABC-MCMC on observed datasets from datagen/gen_observed.py.
  python inference/run_study.py --model 4d --obs data/obs_4d_case1.npz --seeds 1-5 \
      --mode G --iters 3000 --out results/abc_4d_G.npz
`--mode gauss` needs --surrogate results/surrogate_4d.pt.
"""
import argparse, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from sim import settings as S
from inference import mcmc

A = None


def _init(a, D):
    global A
    A = (a, D)


def job(i):
    a, D = A
    rng = np.random.default_rng(1000 + i)
    box = S.BOX_4D if a.model == "4d" else S.BOX_3D
    k = 4 if a.model == "4d" else 3          # 3d = first three paper-order coordinates (delta fixed at 1)
    steps = S.SPEC_2["trans_step"][:k]; th0 = S.SPEC_2["theta_old"][:k]
    lm = np.log10(max(float(D["mom"][i]), 1e-11))
    mu = [lm if m is None else m for m in S.SPEC_2["prior_mu_fixed"]][:k]
    sur = None
    if a.mode == "gauss":
        import train
        sur = train.load(a.surrogate)
    s, acc = mcmc.run(D["obs"][i], box, steps, th0, a.iters, rng, a.mode, (a.h, a.h), a.G,
                      a.model, float(D["tp"]), sur, theta_mu=mu)
    return s, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["3d", "4d"], required=True)
    ap.add_argument("--obs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="1-5"); ap.add_argument("--mode", choices=["G", "gauss"], default="G")
    ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--G", type=int, default=20)
    ap.add_argument("--h", type=float, default=0.01, help="ABC kernel bandwidth on both features")
    ap.add_argument("--surrogate"); ap.add_argument("--cores", type=int, default=None)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    D = dict(np.load(a.obs))
    with Pool(a.cores, initializer=_init, initargs=(a, D)) as p:
        r = p.map(job, range(lo - 1, hi))
    np.savez_compressed(a.out, samples=np.array([x[0] for x in r]), acc=np.array([x[1] for x in r]))
    print("acc rates", [round(x[1], 3) for x in r])


if __name__ == "__main__":
    main()
