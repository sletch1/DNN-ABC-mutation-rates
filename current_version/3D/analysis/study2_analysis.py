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

BURN = 5000
TRUTH = {1: dict(p1=1e-9, p2=1e-8, tau=14.0, delta=0.8), 2: dict(p1=1e-9, p2=5e-8, tau=14.0, delta=0.8)}
NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]


def truth_vec(case, D):
    t = TRUTH[case]; return np.array([np.log10(t["p1"]), np.log10(t["p2"]), t["tau"], np.log10(t["delta"])])[:D]


def load_gps(res, model, case):
    out = {}
    for f in sorted(glob.glob(f"{res}/gps_{model}_case{case}_*.mat")):
        i = int(re.search(r"_(\d+)\.mat$", f).group(1)); m = sio.loadmat(f)
        s = m["sample"]; s = s[:, [1, 2, 3, 0]] if s.shape[1] == 4 else s     # code order (delta,p1,p2,tau) -> paper order
        out[i] = (s[BURN:], float(m["runningTime"].squeeze()))
    return out


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
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="results/tables"); ap.add_argument("--cases", default="1,2")
    a = ap.parse_args(); D = 3 if a.model == "3d" else 4; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs, fid = [], []
    for case in map(int, a.cases.split(",")):
        meth = {"GPS-ABC": load_gps(a.results, a.model, case),
                "DNN-ABC": load_npz(a.results, f"dnnabc_{a.model}_case{case}.npz", BURN),
                "NPE": load_npz(a.results, f"npe_{a.model}_case{case}.npz", 0),
                "ABC-sim": load_npz(a.results, f"abcsim_{a.model}_case{case}.npz", BURN)}
        tv = truth_vec(case, D)
        common = sorted(set.intersection(*[set(v) for v in meth.values() if v])) if meth["GPS-ABC"] else None
        for mname, d in meth.items():
            for i, (draws, secs) in d.items():
                for r in summarise(draws, tv, D):
                    recs.append(dict(model=a.model, case=case, method=mname, dataset=i, secs=secs, in_common=(common is not None and i in common), **r))
        if meth["GPS-ABC"]:                      # fidelity to the GPS-ABC posterior it is meant to replace
            for mname in ("DNN-ABC", "NPE", "ABC-sim"):
                for i in sorted(set(meth[mname]) & set(meth["GPS-ABC"])):
                    for k in range(D):
                        fid.append(dict(model=a.model, case=case, method=mname, dataset=i, param=NAMES[k],
                                        w1=wasserstein_distance(meth[mname][i][0][:, k], meth["GPS-ABC"][i][0][:, k])))
    R = pd.DataFrame(recs); R.to_csv(out / f"recovery_raw_{a.model}.csv", index=False)
    g = R.groupby(["case", "method", "param"])
    T = g.agg(n=("dataset", "nunique"), rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))), bias=("err", "mean"),
              ci_len=("hi", lambda s: np.nan), post_sd=("sd", "mean"), coverage=("cover", "mean"), secs=("secs", "mean")).reset_index()
    T["ci_len"] = g.apply(lambda d: float(np.mean(d.hi - d.lo))).values
    T.to_csv(out / f"recovery_{a.model}.csv", index=False)
    if fid:
        F = pd.DataFrame(fid); F.to_csv(out / f"fidelity_raw_{a.model}.csv", index=False)
        F.groupby(["case", "method", "param"]).w1.agg(["mean", "std", "count"]).reset_index().to_csv(out / f"fidelity_{a.model}.csv", index=False)
    print(T.round(4).to_string())


if __name__ == "__main__":
    main()
