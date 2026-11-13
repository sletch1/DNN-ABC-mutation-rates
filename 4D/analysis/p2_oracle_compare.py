"""Problem 2: interval length means nothing without a reference. Compare each method's posterior with the oracle posterior
(synthetic-likelihood ABC-MCMC, simulator in the loop; inference/p2_oracle.py) on the data sets where the oracle was run.
Per method and parameter: Wasserstein-1 distance to the oracle (parameter units), the ratio of 95% interval lengths (method / oracle),
and the standardised difference of posterior means (in oracle sds). The oracle's own split-half W1 is the Monte Carlo floor.
Writes results/tables/p2_oracle.json and tables/tab_oracle_4d.tex.
    python analysis/p2_oracle_compare.py --results results --gps-horizon 2000
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance as w1
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study2_analysis import load_gps, load_npz, NAMES, BURN_DNN


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--gps-horizon", type=int, default=2000); ap.add_argument("--tex", default="../tables/tab_oracle_4d.tex"); a = ap.parse_args()
    R = Path(a.results); rows = []; floor = []
    for c in (1, 2):
        meth = {"DNN-ABC": load_npz(a.results, f"dnnabc_4d_case{c}.npz", BURN_DNN), "NPE": load_npz(a.results, f"npe_4d_case{c}.npz", 0), "GPS-ABC": load_gps(a.results, "4d", c, a.gps_horizon)}
        for f in sorted((R / "oracle").glob(f"oracle_case{c}_*.npz")):
            i = int(f.stem.split("_")[-1]) + 1; O = np.load(f)["samples"]; h = len(O) // 2
            floor.append([w1(O[:h, k], O[h:, k]) for k in range(4)])
            for m, d in meth.items():
                if i not in d: continue
                S = d[i][0]
                for k in range(4):
                    lo, hi = np.quantile(S[:, k], [.025, .975]); olo, ohi = np.quantile(O[:, k], [.025, .975])
                    rows.append(dict(case=c, idx=i, method=m, param=NAMES[k], w1=float(w1(S[:, k], O[:, k])), len_ratio=float((hi - lo) / (ohi - olo)),
                                     mean_diff_sd=float((S[:, k].mean() - O[:, k].mean()) / O[:, k].std()), oracle_mean_in_ci=bool(lo <= O[:, k].mean() <= hi)))
    floor = np.mean(floor, 0) if floor else None; summ = {}
    for m in ("GPS-ABC", "DNN-ABC", "NPE"):
        for k, nm in enumerate(NAMES):
            r = [x for x in rows if x["method"] == m and x["param"] == nm]
            if r: summ.setdefault(m, {})[nm] = dict(n=len(r), w1=float(np.mean([x["w1"] for x in r])), len_ratio=float(np.mean([x["len_ratio"] for x in r])),
                                                    mean_diff_sd=float(np.mean([abs(x["mean_diff_sd"]) for x in r])), oracle_mean_in_ci=float(np.mean([x["oracle_mean_in_ci"] for x in r])))
    out = dict(rows=rows, summary=summ, mc_floor_w1=None if floor is None else dict(zip(NAMES, map(float, floor))), gps_horizon=a.gps_horizon)
    Path(a.results, "tables").mkdir(exist_ok=True); json.dump(out, open(Path(a.results) / "tables" / "p2_oracle.json", "w"), indent=1)
    L = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Method & Parameter & $W_1$ to oracle & Length ratio & $|\Delta\text{mean}|$ (oracle sd) & Oracle mean in 95\% CI \\", r"\midrule"]
    for m in summ:
        for nm, v in summ[m].items(): L.append(f"{m} & {nm} & {v['w1']:.3f} & {v['len_ratio']:.2f} & {v['mean_diff_sd']:.2f} & {v['oracle_mean_in_ci']:.2f} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; Path(a.tex).write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
