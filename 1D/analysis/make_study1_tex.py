"""Study I tables and summary numbers from 1D/results/tables/*.csv and mcse.md (all numbers in the manuscript's Study I section come from here).
Writes tables/tab1d_{mse,ci,time,cov}.tex and 1D/results/tables/study1_summary.json.    python 1D/analysis/make_study1_tex.py"""
import csv, json, re, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; T = ROOT / "1D/results/tables"; OUT = ROOT / "tables"
rd = lambda f: list(csv.DictReader(open(T / f)))
mse, ci, tm, cov = rd("table1_mse.csv"), rd("table2_cilength.csv"), rd("table3_timing.csv"), rd("table_coverage.csv")
dse = {}
for line in open(T / "mcse.md"):
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) == 9 and re.match(r"^[0-9.e-]+$", c[0]) and re.match(r"^[+-]?[0-9.]+$", c[-1]): dse[(float(c[0]), int(c[1]))] = float(c[-1])
R = int(cov[0]["R"]); ptex = {1e-4: "10^{-4}", 1e-3: "10^{-3}", 1e-2: "10^{-2}"}
def sci(x):
    m, e = f"{float(x):.2e}".split("e"); return f"{m}e{int(e)}"
def pj(r): return f"${ptex[float(r['p'])]}$ & {int(r['J'])}"
def neg(v): return f"$-${abs(v):.2f}" if v < 0 else f"{v:.2f}"
def write(name, rows): (OUT / name).write_text("\n".join(rows) + "\n\\bottomrule\n")
# Table 1
rows = []
for r in mse:
    k = (float(r["p"]), int(r["J"])); g, d = float(r["GPS-ABC"]), float(r["DNN-ABC"]); cell = lambda m, b=False: (r"\textbf{%s (%.2f)}" if b else "%s (%.2f)") % (sci(r[m]), float(r[m + "_nrmse"]))
    rows.append(" & ".join([pj(r), cell("MOM"), cell("MLE"), cell("ABC-MCMC"), cell("GPS-ABC", g < d), cell("DNN-ABC", d <= g), neg(dse[k])]) + r" \\")
write("tab1d_mse.tex", rows)
rows = []
for r in ci:
    g, d = float(r["GPS-ABC"]), float(r["DNN-ABC"]); f = lambda m, b=False: (r"\textbf{%s}" if b else "%s") % sci(r[m])
    rows.append(" & ".join([pj(r), f("ABC-MCMC"), f("GPS-ABC", g < d), f("DNN-ABC", d <= g)]) + r" \\")
write("tab1d_ci.tex", rows)
rows = []
for r in tm: rows.append(" & ".join([pj(r), f"{float(r['ABC-MCMC']):.1f}", f"{float(r['GPS-ABC']):.2f}", f"{float(r['DNN-ABC']):.2f}", r"$%d\times$" % round(float(r["ABC-MCMC"]) / float(r["DNN-ABC"]))]) + r" \\")
write("tab1d_time.tex", rows)
rows = []
for r in cov: rows.append(" & ".join([pj(r)] + [f"{float(r[m + '_coverage']):.3f} ({float(r[m + '_coverage_mcse']):.3f})" for m in ("ABC-MCMC", "GPS-ABC", "DNN-ABC")]) + r" \\")
write("tab1d_cov.tex", rows)
# summary
S = {"R": R}; gl = [(float(r["DNN-ABC"]) < float(r["GPS-ABC"])) for r in mse]; S["n_cells"] = len(mse); S["n_dnn_mse_lower"] = sum(gl)
a = {k: abs(v) for k, v in dse.items()}; kmax = max(a, key=a.get); S["max_abs_dse"] = a[kmax]; S["max_dse_p_exp"] = int(round(__import__("math").log10(kmax[0]))); S["max_dse_J"] = kmax[1]
S["n_resolved"] = sum(v > 2 for v in a.values()); S["n_resolved_favour_gps"] = sum(v < -2 for v in dse.values()); S["n_resolved_favour_dnn"] = sum(v > 2 for v in dse.values())
sh = [100 * (1 - float(r["DNN-ABC"]) / float(r["GPS-ABC"])) for r in ci]; S["n_ci_shorter_than_gps"] = sum(x > 0 for x in sh); S["ci_mean_shrink_pct"] = st.mean(sh); S["ci_max_shrink_pct"] = max(sh); S["ci_min_shrink_pct"] = min(sh)
S["n_ci_narrower_than_abc"] = sum(float(r["DNN-ABC"]) < float(r["ABC-MCMC"]) for r in ci)
sp = [float(r["ABC-MCMC"]) / float(r["DNN-ABC"]) for r in tm]; S["speedup_min"] = min(sp); S["speedup_max"] = max(sp)
S["dnn_s_per_100_min"] = min(float(r["DNN-ABC"]) for r in tm); S["dnn_s_per_100_max"] = max(float(r["DNN-ABC"]) for r in tm)
for m, n in (("ABC-MCMC", "abc"), ("GPS-ABC", "gps"), ("DNN-ABC", "dnn")):
    j10 = [float(r[m + "_coverage"]) for r in cov if int(r["J"]) == 10]; j100 = [float(r[m + "_coverage"]) for r in cov if int(r["J"]) == 100]
    S[f"{n}_cov_j10_min"], S[f"{n}_cov_j10_max"] = min(j10), max(j10); S[f"{n}_cov_j100_min"], S[f"{n}_cov_j100_max"] = min(j100), max(j100)
    S[f"{n}_cov_mean"] = st.mean(float(r[m + "_coverage"]) for r in cov)
S["cov_mcse_median"] = st.median(float(r["DNN-ABC_coverage_mcse"]) for r in cov)
json.dump(S, open(T / "study1_summary.json", "w"), indent=1); print(json.dumps(S, indent=1))
