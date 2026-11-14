"""Collapse results/npe_candidates/*.json into results/npe_candidates_summary.json (scalars per candidate) for revision_numbers.py.
    python 4D/analysis/npe_candidates_summary.py"""
import json, glob, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"; out = {}
for f in sorted(glob.glob(str(R / "npe_candidates" / "*.json"))):
    r = json.load(open(f)); s = r["sbc"]
    out[r["candidate"]] = {"heldout_logq": r["heldout_logq"], "mean_cov95": st.mean(v["cov95"] for v in s.values()), "min_chi2_p": min(v["chi2_p"] for v in s.values()),
                           "contraction_log10_p2": s["log10 p2"]["contraction"], "contraction_log10_delta": s["log10 delta"]["contraction"],
                           "sample_s_500_draws": r["sample_s_per_dataset_L"], "n_designs": r["n_sbc"]}
out["nsf"]["lead_over_maf"] = out["nsf"]["heldout_logq"] - out["maf"]["heldout_logq"]
json.dump(out, open(R / "npe_candidates_summary.json", "w"), indent=1)
