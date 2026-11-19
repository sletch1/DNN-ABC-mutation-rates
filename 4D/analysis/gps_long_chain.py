"""Diagnostics for the 20,000-iteration GPS-ABC chains (data set 1 of each case; results/gps/gps4dL_case*_1.mat).
(1) Convergence of the retained draws (iterations 5,001-20,000): split R-hat over four segments, effective sample size,
    segment-by-segment medians and 95% intervals, acceptance rate.
(2) Where the long chain sits relative to the other methods on the same data set: Wasserstein-1 distance to DNN-ABC, NPE and the
    synthetic-likelihood reference (oracle), and the ratio of 95% interval lengths, parameter by parameter.
Writes results/tables/gps_long_chain.json.   python analysis/gps_long_chain.py --results results
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance as w1
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study2_analysis import NAMES, BURN_DNN, truth_vec, _gps_chain, load_npz


def ess(x):
    """Effective sample size from the autocorrelation function, Geyer's initial positive sequence."""
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]; ac /= ac[0]
    s, t = 0.0, 1
    while t + 1 < n:
        pair = ac[t] + ac[t + 1]
        if pair < 0: break
        s += pair; t += 2
    return n / (1 + 2 * s)


def split_rhat(x, m=4):
    n = len(x) // m; c = np.array([x[i * n:(i + 1) * n] for i in range(m)])
    W = c.var(axis=1, ddof=1).mean(); B = n * c.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--out", default="results/tables/gps_long_chain.json"); a = ap.parse_args()
    out = {}
    for case in (1, 2):
        S, n, el = _gps_chain(f"{a.results}/gps/gps4dL_case{case}_1.mat"); B = S[5000:]          # the paper's burn-in
        truth = truth_vec(case, 4); res = dict(hours=el / 3600, n_iter=int(n), acc=float(np.mean(np.any(np.diff(S, axis=0) != 0, axis=1))), params={})
        others = {"DNN-ABC": load_npz(a.results, f"dnnabc_4d_case{case}.npz", BURN_DNN).get(1), "NPE": load_npz(a.results, f"npe_4d_case{case}.npz", 0).get(1)}
        orc = Path(a.results) / "oracle" / f"oracle_case{case}_0.npz"
        ref = {k: v[0] for k, v in others.items() if v}
        if orc.exists(): ref["Oracle"] = np.load(orc)["samples"]
        for k, nm in enumerate(NAMES):
            x = B[:, k]; seg = np.array_split(x, 4); lo, hi = np.quantile(x, [.025, .975])
            p = dict(rhat=split_rhat(x), ess=float(ess(x)), ess_per_1000_iter=float(1000 * ess(x) / len(x)),
                     seg_median=[float(np.median(s)) for s in seg], seg_len=[float(np.diff(np.quantile(s, [.025, .975]))[0]) for s in seg],
                     median=float(np.median(x)), lo=float(lo), hi=float(hi), truth=float(truth[k]), covers=bool(lo <= truth[k] <= hi), vs={})
            for m, R in ref.items():
                rlo, rhi = np.quantile(R[:, k], [.025, .975])
                p["vs"][m] = dict(w1=float(w1(x, R[:, k])), len_ratio=float((hi - lo) / (rhi - rlo)), mean_diff_sd=float((x.mean() - R[:, k].mean()) / R[:, k].std()))
            res["params"][nm] = p
        out[f"case{case}"] = res
        print(f"\nCase {case}: {n} iterations, {el / 3600:.1f} h, acceptance {res['acc']:.3f}")
        for nm, p in res["params"].items():
            print(f"  {nm:<12} R-hat {p['rhat']:.3f}  ESS {p['ess']:7.0f} ({p['ess_per_1000_iter']:.0f}/1000)  seg medians {np.round(p['seg_median'], 2)}  seg lengths {np.round(p['seg_len'], 2)}")
            for m, v in p["vs"].items(): print(f"      vs {m:<8} W1 {v['w1']:.3f}  length ratio {v['len_ratio']:.2f}  mean diff {v['mean_diff_sd']:+.2f} sd")
    json.dump(out, open(a.out, "w"), indent=1)
    summ = {}                                   # flat scalars for revision_numbers.py -> LaTeX macros
    for c, r in out.items():
        d = {"hours": r["hours"], "acc": r["acc"]}
        for nm, p in r["params"].items():
            d[nm] = dict(rhat=p["rhat"], ess=int(round(p["ess"])), seg_len_min=min(p["seg_len"]), seg_len_max=max(p["seg_len"]))
            if "Oracle" in p["vs"]: d[nm].update(wass_oracle=p["vs"]["Oracle"]["w1"], len_ratio_oracle=p["vs"]["Oracle"]["len_ratio"], mean_diff_oracle_sd=p["vs"]["Oracle"]["mean_diff_sd"])
        d["max_rhat"] = max(p["rhat"] for p in r["params"].values()); summ[c] = d
    json.dump(summ, open(a.out.replace(".json", "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
