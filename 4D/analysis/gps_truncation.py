"""Truncation check for GPS-ABC: one data set per case run to 20,000 iterations (results/gps/gps4dL_case*_1.mat),
compared with the same chain truncated at 2,000 iterations (the horizon used for the 20-data-set comparison).
Same burn-in rule as study2_analysis.py (first quarter discarded).

    python analysis/gps_truncation.py --results results --horizons 2000,20000
"""
import argparse, json
import numpy as np
import scipy.io as sio
from study2_analysis import NAMES, truth_vec, summarise, _gps_chain


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--horizons", default="2000,20000")
    ap.add_argument("--out", default="results/tables/gps_truncation.json"); a = ap.parse_args()
    H = [int(h) for h in a.horizons.split(",")]; out = {}; hours = {}; iters = {}
    for case in (1, 2):
        S, n, el = _gps_chain(f"{a.results}/gps/gps4dL_case{case}_1.mat")
        truth = truth_vec(case, S.shape[1]); out[case] = {}; hours[case] = el / 3600; iters[case] = n
        print(f"\nCase {case}: {n} iterations, {el / 3600:.1f} h, truth {dict(zip(NAMES, truth))}")
        for h in H:
            rows = summarise(S[h // 4:h], truth, S.shape[1]); out[case][h] = rows
            for r in rows:
                print(f"  H={h:>6}  {r['param']:<12} median {r['median']:8.3f}  95% [{r['lo']:8.3f}, {r['hi']:8.3f}]  len {r['hi'] - r['lo']:7.3f}  covers {int(r['cover'])}")
        for k, nm in enumerate(NAMES):
            s, l = out[case][H[0]][k], out[case][H[-1]][k]
            print(f"  shift {nm:<12} median {l['median'] - s['median']:+.3f}  length ratio {(l['hi'] - l['lo']) / (s['hi'] - s['lo']):.2f}")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    summ = {}                                   # flat scalars for revision_numbers.py -> LaTeX macros
    for case in (1, 2):
        c = {"hours": hours[case], "iters": iters[case]}
        for k, nm in enumerate(NAMES):
            s, l = out[case][H[0]][k], out[case][H[-1]][k]
            c[nm] = dict(median_shift=l["median"] - s["median"], length_ratio=(l["hi"] - l["lo"]) / (s["hi"] - s["lo"]), short_len=s["hi"] - s["lo"], long_len=l["hi"] - l["lo"],
                         covers_short=bool(s["cover"]), covers_long=bool(l["cover"]))
        summ[f"case{case}"] = c
    json.dump(summ, open(a.out.replace(".json", "_summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
