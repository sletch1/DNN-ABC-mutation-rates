"""Generate every Study II table of the manuscript as a LaTeX fragment (booktabs) from the result files, so no number is
copied by hand. Output: ../tables/*.tex, pulled into manuscript.tex with \\input.
    python analysis/make_tables_tex.py --results results --out ../tables
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

NM = {"mlp_16-16_tanh": "MLP 16--16 (tanh)", "mlp_32-16_gelu": "MLP 32--16 (GELU)", "mlp_64-32_gelu-tanh": "MLP 64--32 (GELU/tanh)",
      "mlp_128-64_gelu": "MLP 128--64 (GELU)", "mlp_128-64_tanh": "MLP 128--64 (tanh)", "mlp_128-64_silu": "MLP 128--64 (SiLU)",
      "mlp_128-64_relu": "MLP 128--64 (ReLU)", "mlp_256-128-64_gelu": "MLP 256--128--64 (GELU)", "mlp_128x3_silu": "MLP 3$\\times$128 (SiLU)",
      "mlp_64x4_gelu": "MLP 4$\\times$64 (GELU)", "res_128x3_gelu": "ResNet 3$\\times$128 (GELU)", "res_256x4_gelu": "ResNet 4$\\times$256 (GELU)"}
MN = {"DNN-ABC-rho": r"DNN-ABC-$\rho$", "DNN-ABC-rho (1,500)": r"DNN-ABC-$\rho$ (1,500)"}
PN = {"log10 p1": r"$\log_{10}p_1$", "log10 p2": r"$\log_{10}p_2$", "tau": r"$\tau$", "log10 delta": r"$\log_{10}\delta$"}


def pm(m, s, f="{:.3f}"):
    return (f + r"$\,\pm\,$" + f).format(m, s) if np.isfinite(s) else f.format(m)


def tab_arch(res, tag, out):
    r = pd.read_csv(Path(res) / f"arch_search_{tag}_raw.csv"); gp = r[r.arch.str.startswith("GP_")]; d = r[~r.arch.str.startswith("GP_")]
    chosen = json.load(open(Path(res) / f"surrogate_{tag.lower()}_metrics.json"))["arch"]
    g = d.groupby("arch"); m, s = g.mean(numeric_only=True), g.std(numeric_only=True, ddof=1); m = m.sort_values("val_loss")
    rows = []
    for a in m.index:
        b = r"\textbf{%s}" if a == chosen else "%s"
        rows.append(" & ".join([b % NM.get(a, a), f"{int(m.params[a]):,}".replace(",", "{,}"), pm(m.test_f1_x_floor[a], s.test_f1_x_floor[a]),
                                pm(m.test_f2_x_floor[a], s.test_f2_x_floor[a]), f"{m.test_f1_cov95_conf[a]:.3f}", f"{m.test_auc0[a]:.4f}",
                                f"{m.latency_us[a]:.0f}", f"{m.batch1e5_ms[a]:.0f}"]) + r" \\")
    gj = Path(res) / f"gp_baseline_{tag}.json"
    rows.append(r"\midrule")
    for b in (json.load(open(gj)) if gj.exists() else []):
        rows.append(" & ".join([f"GP, {b['budget']:,} points ($\\log_{{10}}\\bar d$ only)".replace(",", "{,}", 1) if False else f"GP, {b['budget']} points ($\\log_{{10}}\\bar d$ only)", "---", f"{b['test_x_floor']:.2f}", "---", f"{b['test_cov95_raw']:.3f}", "---", "---", "---"]) + r" \\")
    hdr = (r"\begin{tabular}{lrrrrrrr}" + "\n" + r"\toprule" + "\n" + r"Surrogate & Params & $\times$ floor $(\log_{10}\bar d)$ & $\times$ floor $(\log_{10}s)$ & Cov.\ 95\% & AUC (zero) & \textmu s/query & ms/$10^5$ \\" + "\n" + r"\midrule")
    (Path(out) / f"tab_arch_{tag.lower()}.tex").write_text(hdr + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")


def tab_recovery(res, tag, out, paired=False):
    f = Path(res) / "tables" / (f"recovery_paired_{tag}.csv" if paired else f"recovery_{tag}.csv")
    if not f.exists(): return
    T = pd.read_csv(f); lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Case & Method & Parameter & RMSE & 95\% CI length & Coverage & s/dataset \\", r"\midrule"]
    for case, dc in T.groupby("case"):
        first = True
        for meth in ["GPS-ABC", "DNN-ABC", "DNN-ABC (1,500)", "DNN-ABC-rho", "DNN-ABC-rho (1,500)", "NPE", "ABC-sim"]:
            dm = dc[dc.method == meth]; dm = dm.assign(_o=dm.param.map({'log10 p1': 0, 'log10 p2': 1, 'tau': 2, 'log10 delta': 3})).sort_values('_o')
            for i, (_, r) in enumerate(dm.iterrows()):
                lines.append(" & ".join([str(case) if (first and i == 0) else "", (MN.get(meth, meth) + f" ($n={int(r.n)}$)") if i == 0 else "", PN[r.param], f"{r.rmse:.3f}",
                                         f"{r.ci_len:.3f}", f"{r.coverage:.2f}", (f"{r.secs:,.1f}" if i == 0 else "")]) + r" \\")
            if len(dm): first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    (Path(out) / (f"tab_recovery_paired_{tag}.tex" if paired else f"tab_recovery_{tag}.tex")).write_text("\n".join(lines) + "\n")


def tab_fidelity(res, tag, out):
    f = Path(res) / "tables" / f"fidelity_{tag}.csv"
    if not f.exists(): return
    F = pd.read_csv(f); F = F[F.method.isin(["DNN-ABC", "DNN-ABC-rho", "NPE"])]; lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"Case & Method & Parameter & $W_1$ to GPS-ABC & $n$ \\", r"\midrule"]
    for case, dc in F.groupby("case"):
        for meth, dm in dc.groupby("method"):
            for i, (_, r) in enumerate(dm.iterrows()):
                lines.append(" & ".join([str(case) if (i == 0 and meth == dc.method.iloc[0]) else "", MN.get(meth, meth) if i == 0 else "", PN[r.param], pm(r["mean"], r["std"] / np.sqrt(r["count"]), "{:.3f}"), str(int(r["count"]))]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    (Path(out) / f"tab_fidelity_{tag}.tex").write_text("\n".join(lines) + "\n")


def tab_sbc(res, tag, out):
    f = Path(res) / f"sbc_{tag}.json"
    if not f.exists(): return
    S = json.load(open(f)); S = {k: v for k, v in S.items() if k in ("DNN-ABC", "NPE")}
    g = Path(res) / f"sbc_{tag}_rho.json"
    if g.exists(): S["DNN-ABC-rho"] = json.load(open(g))["DNN-ABC"]
    order = [m for m in ("DNN-ABC", "DNN-ABC-rho", "NPE") if m in S]
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Method & Parameter & Cov.\ 50\% & Cov.\ 90\% & Cov.\ 95\% & Rank $\chi^2$ $p$ & Contraction \\", r"\midrule"]
    for meth in order:
        for i, (p, v) in enumerate(S[meth].items()):
            lines.append(" & ".join([MN.get(meth, meth) if i == 0 else "", PN[p], f"{v['cov50']:.3f}", f"{v['cov90']:.3f}", f"{v['cov95']:.3f}", f"{v['chi2_p']:.3f}", f"{v['contraction']:.3f}"]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    (Path(out) / f"tab_sbc_{tag}.tex").write_text("\n".join(lines) + "\n")


def tab_arch_supp(res, tag, out):
    r = pd.read_csv(Path(res) / f"arch_search_{tag}_raw.csv"); d = r[~r.arch.str.startswith("GP_")]
    g = d.groupby("arch"); m, sd = g.mean(numeric_only=True), g.std(numeric_only=True, ddof=1); m = m.sort_values("val_loss")
    rows = [" & ".join([NM.get(a, a), f"{int(m.params[a]):,}".replace(",", "{,}"), pm(m.val_loss[a], sd.val_loss[a], "{:.4f}"), f"{m.test_nll[a]:.3f}",
                        f"{m.test_f1_rmse[a]:.4f}", f"{m.test_f2_rmse[a]:.4f}", f"{m.test_ll0[a]:.4f}", f"{m.train_s[a]:.0f}", f"{m.epochs[a]:.0f}"]) + r" \\" for a in m.index]
    hdr = (r"\begin{tabular}{lrrrrrrrr}" + "\n" + r"\toprule" + "\n" + r"Surrogate & Params & Val.\ loss & Test NLL & RMSE $\log_{10}\bar d$ & RMSE $\log_{10}s$ & Zero log-loss & Train s & Epochs \\" + "\n" + r"\midrule")
    (Path(out) / f"tab_arch_supp_{tag.lower()}.tex").write_text(hdr + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")


def tab_validation(res, out):
    f = Path(res) / "validation_vs_published.json"
    if not f.exists(): return
    V = json.load(open(f)); lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Released training set & Values & Zero frac. & Mean & Rep.\ sd & KS $p$ (all) & Smallest bin $p$ \\", r"\midrule"]
    for v in V:
        bp = min(min(x) for x in v["by_param"].values()); pu = v["published"]
        lines.append(" & ".join([v["name"].replace("_", r"\_"), f"{v['n_values']:,}".replace(",", "{,}"), f"{pu['zero_frac']:.3f}", f"{pu['mean']:.4f}", f"{pu['sd_rep']:.4f}", f"{v['ks_all']['p']:.3f}", f"{bp:.3f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (Path(out) / "tab_validation.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--out", default="../tables"); a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for tag in ("4D",):
        if (Path(a.results) / f"arch_search_{tag}_raw.csv").exists(): tab_arch(a.results, tag, a.out); tab_arch_supp(a.results, tag, a.out)
        tab_recovery(a.results, tag.lower(), a.out); tab_recovery(a.results, tag.lower(), a.out, paired=True); tab_fidelity(a.results, tag.lower(), a.out); tab_sbc(a.results, tag.lower(), a.out)
    tab_validation(a.results, a.out)
    print("tables written to", a.out)
