"""Table for the simulation-budget comparison of DNN-ABC (exact sampling) and NPE-Z (inference/p12_budget.py, p12_driver.py).
Every number is read from the result files; nothing is typed.  Means over training seeds; the test set is shared across seeds.
    cd 4D && python analysis/make_budget_table.py --results results/budget_k1000 --out ../tables
"""
import argparse, glob, json
from pathlib import Path
import numpy as np

NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results/budget_k1000"); ap.add_argument("--out", default="../tables"); a = ap.parse_args()
    ns = sorted({int(Path(f).stem.split("_")[1][1:]) for f in glob.glob(f"{a.results}/p12_n*_s*.json")})
    L = [r"\begin{tabular}{rlcccccc}", r"\toprule",
         r"Designs & Method & \multicolumn{4}{c}{Coverage of the $95\%$ interval} & \multicolumn{2}{c}{Contraction} \\", r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
         r" & & $\log_{10}p_1$ & $\log_{10}p_2$ & $\tau$ & $\log_{10}\delta$ & $\log_{10}p_2$ & $\log_{10}\delta$ \\", r"\midrule"]
    info = {}
    for n in ns:
        R = [json.load(open(f)) for f in sorted(glob.glob(f"{a.results}/p12_n{n}_s*.json"))]; K = R[0]["dnn_abc_exact"]["all"]["tau"]["n"]; info[n] = (len(R), K)
        for m, lab in (("dnn_abc_exact", r"DNN-ABC, exact"), ("npe_z", r"NPE-Z")):
            g = lambda k, p: np.mean([r[m]["all"][p][k] for r in R])
            cells = []
            for p in NAMES:
                c = g("cov95", p); z = (c - 0.95) / np.sqrt(0.95 * 0.05 / K); cells.append(f"{c:.3f}" + (r"$^{\ast}$" if abs(z) > 2 else ""))
            L.append(f"{n:,}".replace(",", "{,}").join(["$", "$"]) + f" & {lab} & " + " & ".join(cells) + f" & {g('contraction', 'log10 p2'):.3f} & {g('contraction', 'log10 delta'):.3f} \\\\")
        if n != ns[-1]: L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "tab_budget_4d.tex").write_text("\n".join(L) + "\n")
    seeds, K = info[ns[0]]; (Path(a.out) / "tab_budget_4d_meta.tex").write_text(f"\\newcommand{{\\pbudgetSeeds}}{{{seeds}}}\n\\newcommand{{\\pbudgetK}}{{{K:,}}}\n".replace(f"{K:,}", f"{K:,}".replace(",", "{,}")))
    print("\n".join(L))


if __name__ == "__main__":
    main()
