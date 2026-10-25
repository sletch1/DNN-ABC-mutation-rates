"""Training data as CSV in the same layout as current_work/Models/*/data/slow_data_*.csv:

  2-stage (3d, 4d): Z0,a,delta,p1,p2,tau,tp,J,design,rep,d_bar,d_1..d_J
  1d              : Z0,a,delta,p,tp,J,rep,d_bar,d_1..d_J
One row = one replicate dataset of J independent single-culture draws (Z0=1) at one LHS
design point; d_i=(X_i/Z_i)^(1/4) (the repo's statistic, Z_i==0 redrawn as in getIniX2a.m),
d_bar=mean(d_i). p1,p2,delta,p are on the natural scale (as in the existing CSVs); 3d fixes
delta=1, 4d draws delta in [0.5,2] (log10-uniform via the LHS in dt_range).

  python datagen/gen_training.py --model 4d --n 10000 --rep 5 --out data/slow_data_4D.csv
--exact uses the cell-by-cell simulator (4d: sim.mbp.mut2stage_exact_delta); needs a small --tp.
Per-(design,rep) seeds come from SeedSequence(seed).spawn, so output is independent of --cores.
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import mbp, lhs, settings as S


def _work(args):
    theta, seed, kind, J, tp, exact = args
    rng = np.random.default_rng(seed)
    return [mbp.draw_stat(rng, theta, kind, S.Z0, S.A, tp, S.ROOT, exact) for _ in range(J)]


def _indexed(kv):
    return kv[0], _work(kv[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["1d", "3d", "4d"], required=True)
    ap.add_argument("--n", type=int, default=1500, help="LHS design points")
    ap.add_argument("--rep", type=int, default=None, help="rows per design point (default 10 for 1d, 5 for 2-stage)")
    ap.add_argument("--J", type=int, default=100)
    ap.add_argument("--p", type=int, default=8, help="1d only: p=10^-p sets box and tp")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--tp", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--no-optimal", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.model == "1d":
        box, rep = [S.S1_RANGE[a.p][0]], a.rep or S.SPEC_1["num_rep"]
        tp = a.tp or S.tp_study1(10.0 ** -a.p)
    else:
        box, rep = (S.BOX_3D if a.model == "3d" else S.BOX_4D), a.rep or S.SPEC_2["num_rep"]
        tp = a.tp or S.TP_2
    ss = np.random.SeedSequence(a.seed)
    theta = lhs.design(a.n, box, np.random.default_rng(ss.spawn(1)[0]), optimal=not a.no_optimal)
    seeds = ss.spawn(a.n * rep)
    tasks = [(theta[i], seeds[i * rep + r], a.model, a.J, tp, a.exact) for i in range(a.n) for r in range(rep)]
    t0 = time.time()
    with Pool(a.cores) as pool:
        D = [None] * len(tasks)
        for k, v in pool.imap_unordered(_indexed, enumerate(tasks), chunksize=16):   # dynamic balancing across workers
            D[k] = v

    if a.model == "1d":
        head = ["Z0", "a", "delta", "p", "tp", "J", "rep"]
    else:
        head = ["Z0", "a", "delta", "p1", "p2", "tau", "tp", "J", "design", "rep"]
    head += ["d_bar"] + [f"d_{j}" for j in range(1, a.J + 1)]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(head)
        for k, d in enumerate(D):
            i, r = divmod(k, rep); t = theta[i]
            if a.model == "1d":
                pre = [S.Z0, S.A, 1, 10 ** t[0], tp, a.J, r + 1]
            elif a.model == "3d":
                pre = [S.Z0, S.A, 1, 10 ** t[0], 10 ** t[1], t[2], tp, a.J, i + 1, r + 1]
            else:
                pre = [S.Z0, S.A, 10 ** t[3], 10 ** t[0], 10 ** t[1], t[2], tp, a.J, i + 1, r + 1]
            w.writerow(pre + [np.mean(d)] + list(d))
    print(f"{a.model}: {len(D)} rows ({a.n} designs x {rep} reps, J={a.J}, tp={tp:.4f}, "
          f"{'exact' if a.exact else 'fast'}) -> {a.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
