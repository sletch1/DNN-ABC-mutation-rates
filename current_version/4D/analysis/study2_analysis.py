"""Study II analysis: parameter recovery, interval coverage and length, closeness to GPS-ABC, and cost,
over the observed datasets of each truth case. Reads the chain/draw files written by inference/ and gps_abc/,
writes CSV and LaTeX tables to results/tables/.

Coordinates throughout are the paper's theta = (log10 p1, log10 p2, tau [, log10 delta]).
GPS-ABC .mat files hold the code's order (log10 delta, log10 p1, log10 p2, tau): reordered on load.

    python analysis/study2_analysis.py --model 4d --results results --out results/tables
"""
import argparse, glob, json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance
import scipy.io as sio

BURN = 5000          # GPS-ABC: the paper's burn-in
BURN_DNN = 15000     # DNN-ABC / simulator-ABC chains: 10,000 adaptation + 5,000 burn-in
TRUTH = {1: dict(p1=1e-9, p2=1e-8, tau=14.0, delta=0.8), 2: dict(p1=1e-9, p2=5e-8, tau=14.0, delta=0.8)}
NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]


def truth_vec(case, D):
    t = TRUTH[case]; return np.array([np.log10(t["p1"]), np.log10(t["p2"]), t["tau"], np.log10(t["delta"])])[:D]


def _gps_chain(f):
    """Final result file or a checkpoint -> (chain in paper order, iterations done, seconds elapsed)."""
    m = sio.loadmat(f)
    if "ss" in m:                                   # checkpoint written by gps_abc/patched/ABC_mu2a.m
        n = int(m["ss"].squeeze()); S, el = m["Sample"][:n], float(m["elapsed"].squeeze())
    else:
        S, el = m["sample"], float(m["runningTime"].squeeze()); n = len(S)
    return (S[:, [1, 2, 3, 0]] if S.shape[1] == 4 else S), n, el


def load_gps(res, model, case, horizon=None):
    """GPS-ABC draws from results/gps/: finished runs (gps4d_case*_i.mat) or checkpoints (ckpt_case*_i.mat). Every dataset is analysed at the
    same chain length H (default: the largest multiple of 250 that every available run has reached) with the paper's burn-in
    fraction (first quarter discarded). Seconds are extrapolated linearly in iterations from the run's elapsed time."""
    runs = {}
    for f in sorted(glob.glob(f"{res}/gps/gps4d_case{case}_*.mat")) + sorted(glob.glob(f"{res}/gps/ckpt_case{case}_*.mat")):
        i = int(re.search(r"_(\d+)\.mat$", f).group(1))
        if i not in runs or "gps4d" in f: runs[i] = _gps_chain(f)
    if not runs: return {}
    H = horizon or int(min(r[1] for r in runs.values()) // 250 * 250)
    return {i: (S[H // 4:H], el * H / n) for i, (S, n, el) in runs.items() if n >= H and H > 0}


def load_npz(res, name, burn):
    f = Path(res) / name
    if not f.exists(): return {}
    z = np.load(f); S = z["samples"]; secs = z["secs"] if "secs" in z else np.full(len(S), float(z["sample_s"]) if "sample_s" in z else np.nan)
    return {i + 1: (S[i][burn:].astype(float), float(secs[i])) for i in range(len(S))}


def summarise(draws, truth, D):
    rows = []
    for k in range(D):
        v = draws[:, k]; lo, hi = np.quantile(v, [.025, .975])
        rows.append(dict(param=NAMES[k], median=np.median(v), mean=v.mean(), sd=v.std(), lo=lo, hi=hi, cover=float(lo <= truth[k] <= hi), err=np.median(v) - truth[k]))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=["3d", "4d"], required=True)
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="results/tables"); ap.add_argument("--cases", default="1,2"); ap.add_argument("--gps-horizon", type=int, default=None, help="GPS-ABC chain length analysed (default: what every run has reached)")
    a = ap.parse_args(); D = 3 if a.model == "3d" else 4; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs, fid = [], []
    for case in map(int, a.cases.split(",")):
        meth = {"GPS-ABC": load_gps(a.results, a.model, case, a.gps_horizon),
                "DNN-ABC": load_npz(a.results, f"dnnabc_{a.model}_case{case}.npz", BURN_DNN),
                "DNN-ABC (1,500)": load_npz(a.results, f"dnnabc1500_{a.model}_case{case}.npz", BURN_DNN),
                "DNN-ABC-rho": load_npz(a.results, f"dnnabcrho_{a.model}_case{case}.npz", BURN_DNN),
                "DNN-ABC-rho (1,500)": load_npz(a.results, f"dnnabcrho1500_{a.model}_case{case}.npz", BURN_DNN),
                "NPE": load_npz(a.results, f"npe_{a.model}_case{case}.npz", 0),
                "ABC-sim": load_npz(a.results, f"abcsim_{a.model}_case{case}.npz", BURN)}
        tv = truth_vec(case, D)
        common = sorted(set.intersection(*[set(v) for v in meth.values() if v])) if meth["GPS-ABC"] else None   # datasets every method has
        for mname, d in meth.items():
            for i, (draws, secs) in d.items():
                for r in summarise(draws, tv, D):
                    recs.append(dict(model=a.model, case=case, method=mname, dataset=i, secs=secs, in_common=(common is not None and i in common), **r))
        if meth["GPS-ABC"]:                      # fidelity to the GPS-ABC posterior it is meant to replace
            for mname in ("DNN-ABC", "DNN-ABC-rho", "DNN-ABC (1,500)", "DNN-ABC-rho (1,500)", "NPE", "ABC-sim"):
                for i in sorted(set(meth[mname]) & set(meth["GPS-ABC"])):
                    for k in range(D):
                        fid.append(dict(model=a.model, case=case, method=mname, dataset=i, param=NAMES[k],
                                        w1=wasserstein_distance(meth[mname][i][0][:, k], meth["GPS-ABC"][i][0][:, k])))
    R = pd.DataFrame(recs); R.to_csv(out / f"recovery_raw_{a.model}.csv", index=False)
    def aggregate(R):
        g = R.groupby(["case", "method", "param"])
        T = g.agg(n=("dataset", "nunique"), rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))), bias=("err", "mean"), post_sd=("sd", "mean"),
                  coverage=("cover", "mean"), secs=("secs", "mean")).reset_index()
        T["ci_len"] = g.apply(lambda d: float(np.mean(d.hi - d.lo))).values
        return T
    T = aggregate(R); T.to_csv(out / f"recovery_{a.model}.csv", index=False)                       # every dataset each method was run on
    if R.in_common.any():                                                                          # paired: only datasets GPS-ABC also ran
        aggregate(R[R.in_common]).to_csv(out / f"recovery_paired_{a.model}.csv", index=False)
    if fid:
        F = pd.DataFrame(fid); F.to_csv(out / f"fidelity_raw_{a.model}.csv", index=False)
        F.groupby(["case", "method", "param"]).w1.agg(["mean", "std", "count"]).reset_index().to_csv(out / f"fidelity_{a.model}.csv", index=False)
    print(T.round(4).to_string())


if __name__ == "__main__":
    main()
