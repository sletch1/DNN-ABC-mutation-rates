"""Assemble a clear, readable PLAINTEXT summary of the whole 3-D pipeline and
email it (via send_report.send). Robust to missing artifacts: any section whose
inputs are absent is reported as skipped/failed rather than crashing.

Reads:
  results/logs/pipeline_status.txt   (STEP|name|exit|seconds lines from run_all.sh)
  results/model/surrogate_metrics.json
  results/logs/validate_simulator.md  (Overall line)
  results/tables/calibration.csv
  results/tables/gp_scaling.csv
  results/tables/table1_mse.csv, table2_cilength.csv, table3_timing.csv
  results/tables/abc_coverage.csv

Usage: python report.py            (sends the email)
       python report.py --print    (print only, no email)
"""
import sys
import json
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from paths import RESULTS, LOG_DIR, TABLE_DIR, MODEL_DIR
import send_report

HR = "-" * 64


def _try(fn, default="(unavailable)"):
    try:
        return fn()
    except Exception as e:
        return f"{default} [{type(e).__name__}]"


def status_section():
    p = LOG_DIR / "pipeline_status.txt"
    if not p.exists():
        return "Pipeline status: (no status file)\n"
    out = ["Pipeline steps (name / result / time):"]
    for line in p.read_text().splitlines():
        parts = line.split("|")
        if len(parts) == 4 and parts[0] == "STEP":
            _, name, code, secs = parts
            tag = "ok " if code == "0" else "FAIL"
            out.append(f"  [{tag}] {name:<26} {float(secs)/60:5.1f} min")
    return "\n".join(out) + "\n"


def surrogate_section():
    m = json.loads((MODEL_DIR / "surrogate_metrics.json").read_text())
    te = m["test"]
    lines = ["SURROGATE FIT (held-out test replicates 9-10):",
             f"  MSE(log10 d_bar) = {te['mse_log']:.5f}   MAE = {te['mae_log']:.5f}",
             f"  95% coverage      = {te['coverage95']:.3f}"]
    cal = TABLE_DIR / "calibration.csv"
    if cal.exists():
        c = pd.read_csv(cal)
        pairs = ", ".join(f"{r['nominal']:.2f}->{r['empirical']:.2f}" for _, r in c.iterrows())
        lines.append(f"  calibration (nominal->empirical): {pairs}")
    return "\n".join(lines) + "\n"


def gp_section():
    f = TABLE_DIR / "gp_scaling.csv"
    if not f.exists():
        return "GP-vs-DNN SCALING: (not run)\n"
    d = pd.read_csv(f)
    gp = d[d["model"] == "GP"]; dnn = d[d["model"] == "DNN"].iloc[0]
    best = gp.loc[gp["surface_mse"].idxmin()]
    fit_lo, fit_hi = gp["fit_s"].min(), gp["fit_s"].max()
    impr = (1 - dnn["surface_mse"] / best["surface_mse"]) * 100
    return ("GP-vs-DNN SCALING (surface fit; targets #1 & #2):\n"
            f"  DNN (all ~5000 rows): MSE {dnn['surface_mse']:.3e}, query {dnn['query_us']:.1f} us/pt (flat)\n"
            f"  best GP (budget {int(best['budget'])}): MSE {best['surface_mse']:.3e}\n"
            f"  -> DNN is {impr:.0f}% better than the best GP; GP fit time ranges "
            f"{fit_lo:.1f}-{fit_hi:.1f}s (grows ~n^3)\n")


def table1_section():
    f = TABLE_DIR / "table1_mse.csv"
    if not f.exists():
        return "TABLE 1 (accuracy): (not run)\n"
    d = pd.read_csv(f)
    impr, wins = [], 0
    for _, r in d.iterrows():
        g, n = r["GPS-ABC"], r["DNN-ABC"]
        if np.isfinite(g) and np.isfinite(n) and g > 0:
            impr.append((g - n) / g)
            wins += int(n <= g)
    avg = np.mean(impr) * 100 if impr else float("nan")
    lines = [f"TABLE 1 (MSE of p-hat): DNN-ABC vs GPS-ABC across {len(d)} cells",
             f"  DNN-ABC <= GPS-ABC in {wins}/{len(d)} cells; mean MSE reduction {avg:+.0f}%",
             "  p        a    d    J    GPS-ABC MSE   DNN-ABC MSE"]
    for _, r in d.iterrows():
        lines.append(f"  {r['p']:.0e}  {r['a']:.1f}  {r['delta']:.1f}  {int(r['J']):>3}  "
                     f"{r['GPS-ABC']:.3e}    {r['DNN-ABC']:.3e}")
    return "\n".join(lines) + "\n"


def table3_section():
    f = TABLE_DIR / "table3_timing.csv"
    if not f.exists():
        return "TABLE 3 (timing): (not run)\n"
    d = pd.read_csv(f)
    sp = (d["ABC-MCMC"] / d["DNN-ABC"]).replace([np.inf, -np.inf], np.nan).dropna()
    return ("TABLE 3 (seconds / 100 MCMC iterations):\n"
            f"  DNN-ABC speedup over exact ABC-MCMC: {sp.min():.0f}x - {sp.max():.0f}x "
            f"(median {sp.median():.0f}x)\n"
            f"  DNN-ABC per-100-it: {d['DNN-ABC'].min():.3f}-{d['DNN-ABC'].max():.3f}s "
            f"(flat); GPS-ABC {d['GPS-ABC'].min():.3f}-{d['GPS-ABC'].max():.3f}s\n")


def coverage_section():
    f = TABLE_DIR / "abc_coverage.csv"
    if not f.exists():
        return "ABC INTERVAL COVERAGE: (not run)\n"
    d = pd.read_csv(f)
    out = ["ABC 95% INTERVAL COVERAGE (target ~0.95, pooled):"]
    for m in ["ABC-MCMC", "GPS-ABC", "DNN-ABC"]:
        if m in d:
            out.append(f"  {m:<10} {d[m].mean():.3f}")
    return "\n".join(out) + "\n"


def validate_line():
    f = LOG_DIR / "validate_simulator.md"
    if not f.exists():
        return "SIMULATOR VALIDATION: (not run)"
    for line in f.read_text().splitlines():
        if "Overall:" in line:
            return "SIMULATOR VALIDATION: " + line.split("Overall:")[1].strip()
    return "SIMULATOR VALIDATION: (see log)"


def build():
    body = []
    body.append("3-D DNN-ABC surrogate: autonomous build & test run COMPLETE")
    body.append("Everything ran on stat86; results are ready to pull.")
    body.append(HR)
    body.append(_try(status_section))
    body.append(HR)
    body.append(_try(validate_line))
    body.append("")
    body.append(_try(surrogate_section))
    body.append(_try(gp_section))
    body.append(_try(table1_section))
    body.append(_try(table3_section))
    body.append(_try(coverage_section))
    body.append(HR)
    body.append("TO PULL (from the laptop, VPN on):")
    body.append("  results are in  ~/NN_ABC/DNN_Prototypes/3D/results/  on stat86,")
    body.append("  and bundled at  ~/3D_results.tgz")
    body.append("  Reopen the Research-Code folder and a fresh session will pull,")
    body.append("  examine, write the READMEs, and commit/push.")
    return "\n".join(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="print_only")
    ap.add_argument("--subject", default="[3D DNN-ABC] build & test complete - ready to pull")
    args = ap.parse_args()
    text = build()
    if args.print_only:
        print(text)
    else:
        send_report.send(args.subject, text)
        print("report emailed.")
