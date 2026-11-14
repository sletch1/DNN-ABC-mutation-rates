"""Collapse results/npe_candidates/*.json (one file per candidate and training seed) into results/npe_candidates_summary.json:
means over training seeds, the seed standard deviation of the held-out log-probability, and how often the first candidate beats the second.
    python 4D/analysis/npe_candidates_summary.py"""
import json, glob, statistics as st, collections
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"; runs = collections.defaultdict(dict)
for f in sorted(glob.glob(str(R / "npe_candidates" / "*.json"))):
    r = json.load(open(f)); runs[r["candidate"]][r.get("seed", 0)] = r
out = {}
for c, rs in runs.items():
    L = list(rs.values()); s = lambda g: st.mean(g(r) for r in L)
    lq = [r["heldout_logq"] for r in L]
    out[c] = {"heldout_logq": st.mean(lq), "heldout_logq_sd": st.stdev(lq) if len(lq) > 1 else 0.0, "n_seeds": len(L),
              "mean_cov95": s(lambda r: st.mean(v["cov95"] for v in r["sbc"].values())), "min_chi2_p": s(lambda r: min(v["chi2_p"] for v in r["sbc"].values())),
              "contraction_log10_p2": s(lambda r: r["sbc"]["log10 p2"]["contraction"]), "contraction_log10_delta": s(lambda r: r["sbc"]["log10 delta"]["contraction"]),
              "sample_s_500_draws": s(lambda r: r["sample_s_per_dataset_L"]), "n_designs": L[0]["n_sbc"]}
common = sorted(set(runs["nsf"]) & set(runs["maf"]))
out["nsf"]["lead_over_maf"] = st.mean(runs["nsf"][k]["heldout_logq"] - runs["maf"][k]["heldout_logq"] for k in common)
out["nsf"]["seeds_nsf_beats_maf"] = sum(runs["nsf"][k]["heldout_logq"] > runs["maf"][k]["heldout_logq"] for k in common)
others = [c for c in runs if c not in ("nsf", "maf")]
out["nsf"]["lead_over_third"] = out["nsf"]["heldout_logq"] - max(out[c]["heldout_logq"] for c in others)
json.dump(out, open(R / "npe_candidates_summary.json", "w"), indent=1)
