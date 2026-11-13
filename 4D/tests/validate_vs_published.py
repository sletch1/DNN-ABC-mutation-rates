"""Is our data generator the original authors' generator? A formal distributional test against the training sets they released
(Original_Paper_Code/training/*.mat; local only, not redistributed).

At each of their design points theta_i they ran the simulator R times and stored Y = (X/Z)^(1/4) (single culture, Z0 = 1).
For every stored value y we draw m = 400 values from OUR simulator at the same theta_i, compute the randomised probability-integral
transform  u = F(y-) + V (F(y) - F(y-)),  V ~ U(0,1)  (Y has an atom at 0, so ties must be randomised). If our simulator
has the same distribution at every theta_i, all u are i.i.d. U(0,1); a Kolmogorov--Smirnov test checks this, overall and within
bins of each parameter. Reported with the replicate-level zero fraction, mean and standard deviation.

    python tests/validate_vs_published.py [--m 400] [--cores N] [--out results/validation_vs_published.json]
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, scipy.io as sio
from scipy.stats import kstest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sim import mbp, settings as S
T = ROOT.parent / "Original_Paper_Code" / "training"


def _pit(args):
    th, ys, kind, tp, m, seed = args
    rng = np.random.default_rng(seed)
    d = np.array([mbp.draw_stat(rng, th, kind, 1, 1.0, tp, 4, False) for _ in range(m)])
    d.sort(); u = []
    for y in ys:
        lo = np.searchsorted(d, y, "left") / m; hi = np.searchsorted(d, y, "right") / m
        u.append(lo + rng.random() * (hi - lo))
    return u


def run(theta, Y, kind, tp, m, cores):
    tasks = [(theta[i], Y[i], kind, tp, m, 1000 + i) for i in range(len(theta))]
    with Pool(cores) as p: U = np.array(p.map(_pit, tasks, chunksize=8))
    return U


def report(name, theta, Y, U, names, box):
    out = dict(name=name, n_points=len(theta), n_values=int(U.size), published=dict(zero_frac=float((Y == 0).mean()), mean=float(Y.mean()), sd_rep=float(Y.std(1, ddof=1).mean())),
               ks_all=dict(zip(("stat", "p"), map(float, kstest(U.ravel(), "uniform")))), by_param={})
    for k, nm in enumerate(names):
        lo, hi = box[k]; b = np.minimum(((theta[:, k] - lo) / (hi - lo) * 4).astype(int), 3)
        out["by_param"][nm] = [float(kstest(U[b == q].ravel(), "uniform").pvalue) for q in range(4)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--m", type=int, default=400); ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "validation_vs_published.json")); a = ap.parse_args(); res = []
    if not (T / "init_2a.mat").exists(): sys.exit(f"{T} not found: this check needs the authors' released training sets")
    m = sio.loadmat(T / "init_2a.mat"); th = m["theta_list"][:, [1, 2, 3, 0]]; Y = m["X"][:, 0, :]     # shipped order (delta,p1,p2,tau) -> paper order
    res.append(report("4-parameter (init_2a.mat)", th, Y, run(th, Y, "4d", 20.0, a.m, a.cores), ["log10 p1", "log10 p2", "tau", "log10 delta"], S.BOX_4D))
    for p in (8, 6, 4):
        m = sio.loadmat(T / f"simu1/init1_p{p}.mat"); th1 = m["theta_list"][:, :1]; Y1 = m["X"][:, 0, :]; box = S.S1_RANGE[p][0]
        res.append(report(f"1-parameter p=1e-{p} (init1_p{p}.mat)", th1, Y1, run(th1, Y1, "1d", S.tp_study1(10.0 ** -p), a.m, a.cores), ["log10 p"], [box]))
    for r in res: print(json.dumps(r, indent=1))
    Path(a.out).parent.mkdir(exist_ok=True, parents=True); json.dump(res, open(a.out, "w"), indent=1)
