"""Problem 6: a stronger check that our generator is the original authors'. Extends validate_vs_published.py in four ways:
  * more draws per design point (m = 4000 instead of 400) so our empirical CDF is not the noisy side of the comparison;
  * several seeds (simulator and tie-breaking), reporting every seed's KS p-value, not the best one;
  * moment and zero-fraction comparisons per design point, standardised (z-scores whose mean should be 0 and sd 1);
  * effect sizes (KS distance, relative differences of zero fraction / mean / sd with bootstrap CIs), because a p-value of 0.08 on
    7,500 values says little about how large a difference could be hiding.
A non-significant test is not evidence of equality; the claim made in the paper is the measured difference with its interval.
    python tests/validate_extended.py [--m 4000] [--seeds 5] [--out results/validation_extended.json]
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, scipy.io as sio
from scipy.stats import kstest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "inference"))
from sim import mbp, settings as S
from dyncores import wait_for_core, wide_pool_size
T = ROOT.parent / "Original_Paper_Code" / "training"


def _one(args):
    th, ys, kind, tp, m, seed = args
    wait_for_core(jitter=1, poll=5)
    rng = np.random.default_rng(seed)
    d = np.array([mbp.draw_stat(rng, th, kind, 1, 1.0, tp, 4, False) for _ in range(m)]); d.sort()
    u = [(np.searchsorted(d, y, "left") + rng.random() * (np.searchsorted(d, y, "right") - np.searchsorted(d, y, "left"))) / m for y in ys]
    return u, float((d == 0).mean()), float(d.mean()), float(d.var(ddof=1))


def boot(x, f, B=2000, seed=0):
    r = np.random.default_rng(seed); n = len(x)
    v = [f(x[r.integers(0, n, n)]) for _ in range(B)]; return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def analyse(name, theta, Y, kind, tp, m, seeds):
    out = dict(name=name, n_points=len(theta), R=Y.shape[1], m=m, seeds=[])
    pz_their = (Y == 0).mean(1); mean_their = Y.mean(1)
    for sd in range(seeds):
        with Pool(wide_pool_size(28)) as p:
            res = p.map(_one, [(theta[i], Y[i], kind, tp, m, 10_000 * (sd + 1) + i) for i in range(len(theta))], chunksize=8)
        U = np.array([r[0] for r in res]); pz = np.array([r[1] for r in res]); mu = np.array([r[2] for r in res]); va = np.array([r[3] for r in res])
        R = Y.shape[1]; ks = kstest(U.ravel(), "uniform")
        zz = (pz_their - pz) / np.sqrt(np.maximum(pz * (1 - pz), 1e-12) * (1 / R + 1 / m)); ok = (pz > 0) & (pz < 1)
        zm = (mean_their - mu) / np.sqrt(np.maximum(va, 1e-24) * (1 / R + 1 / m))
        out["seeds"].append(dict(seed=sd, ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
            zero_frac=dict(theirs=float(pz_their.mean()), ours=float(pz.mean()), z_mean=float(zz[ok].mean()), z_sd=float(zz[ok].std())),
            mean=dict(theirs=float(mean_their.mean()), ours=float(mu.mean()), rel_diff=float((mean_their.mean() - mu.mean()) / mu.mean()), z_mean=float(zm.mean()), z_sd=float(zm.std())),
            pit_decile_counts=np.histogram(U.ravel(), bins=10, range=(0, 1))[0].tolist(),
            pit_decile_max_rel_dev=float(np.max(np.abs(np.histogram(U.ravel(), bins=10, range=(0, 1))[0] / (U.size / 10) - 1)))))
        if sd == 0:
            d = (pz_their - pz); out["zero_frac_diff_ci95"] = boot(d, np.mean)
            out["mean_rel_diff_ci95"] = boot(mean_their - mu, lambda x: x.mean() / mu.mean())
            out["quantile_check"] = {str(q): dict(theirs=float(np.quantile(Y[Y > 0], q)), ours_pit_implied="see pit_decile_counts") for q in (.25, .5, .75, .9)}
    ps = [s["ks_p"] for s in out["seeds"]]; out["ks_p_min"], out["ks_p_median"], out["ks_p_max"] = float(min(ps)), float(np.median(ps)), float(max(ps))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--m", type=int, default=4000); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "validation_extended.json")); ap.add_argument("--only4d", action="store_true"); a = ap.parse_args()
    m4 = sio.loadmat(T / "init_2a.mat"); th = m4["theta_list"][:, [1, 2, 3, 0]]; Y = m4["X"][:, 0, :]
    res = [analyse("4-parameter (init_2a.mat)", th, Y, "4d", 20.0, a.m, a.seeds)]; print(json.dumps(res[-1], indent=1), flush=True)
    if not a.only4d:
        for p in (8, 6, 4):
            m = sio.loadmat(T / f"simu1/init1_p{p}.mat"); res.append(analyse(f"1-parameter p=1e-{p}", m["theta_list"][:, :1], m["X"][:, 0, :], "1d", S.tp_study1(10.0 ** -p), a.m, a.seeds))
            print(json.dumps(res[-1], indent=1), flush=True)
    Path(a.out).parent.mkdir(exist_ok=True, parents=True); json.dump(res, open(a.out, "w"), indent=1)
