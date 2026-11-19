"""Tables for (a) the repairs of the overconfident Markov chain (exact importance resampling, Hamiltonian Monte Carlo,
importance-corrected NPE), (b) the epistemic-error test (bootstrap ensemble of surrogates) and (c) the comparison with GPS-ABC
at the original 20,000 iterations. Every number is read from a result file; nothing is typed.
    cd 4D && python analysis/make_repair_tables.py --results results --out ../tables
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study2_analysis import NAMES, BURN_DNN, truth_vec, _gps_chain, load_npz

PN = {"log10 p1": r"$\log_{10}p_1$", "log10 p2": r"$\log_{10}p_2$", "tau": r"$\tau$", "log10 delta": r"$\log_{10}\delta$"}


def cov_cell(c, z=None):
    """Coverage with a dagger when more than two binomial standard errors from 0.95 are exceeded."""
    s = f"{c:.3f}"
    return s + (r"$^{\ast}$" if z is not None and abs(z) > 2 else "")


def zof(c, n):
    return (c - 0.95) / np.sqrt(0.95 * 0.05 / n)


def tab_repairs(R, out):
    sbc = json.load(open(R / "sbc_4d.json")); n364 = int(sbc["n_datasets"]["dbar_positive"])
    isc = json.load(open(R / "p3/is_check_simulator.json")); hmc = json.load(open(R / "p9/hmc_check.json")); nis = json.load(open(R / "p10/npe_is.json"))
    rows = []

    def row(label, cov, contr_p2, n):
        cells = [cov_cell(cov[p], zof(cov[p], n)) for p in NAMES]
        rows.append(" & ".join([label] + cells + [f"{contr_p2:.3f}"]) + r" \\")

    row("DNN-ABC, random-walk chain (as in Table~\\ref{tab:sbc4D})", {p: sbc["DNN-ABC"][p]["cov95"] for p in NAMES}, sbc["DNN-ABC"]["log10 p2"]["contraction"], n364)
    row("DNN-ABC, exact importance resampling", {p: isc[p]["cov95_dbar_pos"] for p in NAMES}, isc["contraction"]["log10 p2"], n364)
    row(r"DNN-ABC, Hamiltonian Monte Carlo$^{\dagger}$", {p: hmc["dbar>0"][p]["cov95"] for p in NAMES}, hmc["contraction"]["log10 p2"], n364)
    rows.append(r"\midrule")
    row("NPE (neural spline flow)", {p: sbc["NPE"][p]["cov95"] for p in NAMES}, sbc["NPE"]["log10 p2"]["contraction"], n364)
    ni = nis["npe_importance_corrected"]; npl = nis["npe_plain"]
    row("NPE, plain (the draws corrected below)", {p: npl[p]["cov95"] for p in NAMES}, npl["log10 p2"]["contraction"], n364)
    row("NPE, importance-corrected by the surrogate likelihood", {p: ni[p]["cov95"] for p in NAMES}, ni["log10 p2"]["contraction"], n364)
    hdr = (r"\begin{tabular}{lccccc}" + "\n" + r"\toprule" + "\n" +
           r"Sampler & \multicolumn{4}{c}{Coverage of the $95\%$ interval} & Contraction \\" + "\n" +
           r"\cmidrule(lr){2-5}" + "\n" + r" & " + " & ".join(PN[p] for p in NAMES) + r" & $\log_{10}p_2$ \\" + "\n" + r"\midrule")
    (out / "tab_repairs_4d.tex").write_text(hdr + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")


LAB = {"single (deployed)": "Deployed network", "E-Lik (ensemble mean likelihood)": "Ensemble of 10, mean likelihood",
       "E-LogLik (ensemble mean log-likelihood)": "Ensemble of 10, mean log-likelihood", "ensemble member 0 alone": "One bootstrap member"}


def tab_epistemic(R, out):
    e = json.load(open(R / "p11/ensemble_is.json")); rows = []
    for v, d in e["scores"].items():
        a = d["dbar>0"]; n = a["n"]
        cells = [cov_cell(a[p]["cov95"], a[p]["z"]) for p in NAMES]
        rows.append(" & ".join([LAB.get(v, v)] + cells + [f"{a['log10 p2']['contraction']:.3f}", f"{e['ess_median'][v]:,.0f}".replace(",", "{,}")]) + r" \\")
    hdr = (r"\begin{tabular}{lccccrr}" + "\n" + r"\toprule" + "\n" + r"Surrogate likelihood & \multicolumn{4}{c}{Coverage of the $95\%$ interval} & Contr. & Median \\" + "\n" +
           r"\cmidrule(lr){2-5}" + "\n" + r" & " + " & ".join(PN[p] for p in NAMES) + r" & $\log_{10}p_2$ & ESS \\" + "\n" + r"\midrule")
    (out / "tab_epistemic_4d.tex").write_text(hdr + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")


def tab_gps_full(R, out):
    rows = []
    for case in (1, 2):
        S, n, el = _gps_chain(str(R / "gps" / f"gps4dL_case{case}_1.mat")); B = S[5000:]; truth = truth_vec(case, 4)
        M = {"GPS-ABC": B, "DNN-ABC": load_npz(str(R), f"dnnabc_4d_case{case}.npz", BURN_DNN).get(1), "NPE": load_npz(str(R), f"npe_4d_case{case}.npz", 0).get(1)}
        M = {k: (v if k == "GPS-ABC" else v[0]) for k, v in M.items() if v is not None}
        orc = R / "oracle" / f"oracle_case{case}_0.npz"
        if orc.exists(): M["Reference"] = np.load(orc)["samples"]
        for k, nm in enumerate(NAMES):
            cells = []
            for m in ("GPS-ABC", "DNN-ABC", "NPE", "Reference"):
                if m not in M: cells.append("---"); continue
                x = M[m][:, k]; lo, hi = np.quantile(x, [.025, .975]); miss = not (lo <= truth[k] <= hi)
                cells.append(f"{np.median(x):.2f} ({hi - lo:.2f})" + (r"$^{\ast}$" if miss else ""))
            lead = f"{case}" if k == 0 else ""
            rows.append(" & ".join([lead, PN[nm], f"{truth[k]:.2f}"] + cells) + r" \\")
        if case == 1: rows.append(r"\midrule")
    hdr = (r"\begin{tabular}{llrcccc}" + "\n" + r"\toprule" + "\n" + r"Case & Parameter & Truth & GPS-ABC, $20{,}000$ it. & DNN-ABC & NPE & Reference \\" + "\n" + r"\midrule")
    rows = [re.sub(r"(?<![-\w])-(?=\d)", "$-$", r) for r in rows]
    (out / "tab_gps_full_4d.tex").write_text(hdr + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--out", default="../tables"); a = ap.parse_args()
    R, O = Path(a.results), Path(a.out)
    for f in (tab_repairs, tab_epistemic, tab_gps_full):
        try: f(R, O); print("wrote", f.__name__)
        except FileNotFoundError as ex: print("skipped", f.__name__, "-", ex)
