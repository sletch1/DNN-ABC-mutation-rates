"""Summaries of the Problem 7 and extended-validation JSONs (means over seeds) for revision_numbers.py.  python 4D/analysis/p7_val_summary.py"""
import json, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
p = json.load(open(R / "p7_extra_features.json")); out = {"n_train": p["n_train"], "n_test": p["n_test"], "seeds": p["seeds"]}
for key, tag in (("(dbar, s)", "base"), ("(dbar, s) + extras", "extras")):
    for n in ("log10 p1", "log10 p2", "tau", "log10 delta"):
        for m in ("contraction", "cov95", "chi2_p"): out[f"{tag} {n} {m}"] = st.mean(x[n][m] for x in p[key])
    out[f"{tag} heldout_logq"] = st.mean(x["heldout_logq"] for x in p[key])
json.dump(out, open(R / "p7_summary.json", "w"), indent=1)
v = json.load(open(R / "validation_extended.json")); s = {}
for x in v:
    k = {"4-parameter (init_2a.mat)": "four", "1-parameter p=1e-8": "oneE8", "1-parameter p=1e-6": "oneE6", "1-parameter p=1e-4": "oneE4"}[x["name"]]; S = x["seeds"]
    s[k] = {"n_points": x["n_points"], "m": x["m"], "seeds": len(S), "ks_p_min": min(a["ks_p"] for a in S), "ks_p_median": st.median(a["ks_p"] for a in S),
            "ks_p_max": max(a["ks_p"] for a in S), "mean_rel_diff_pct": 100 * st.mean(a["mean"]["rel_diff"] for a in S),
            "zero_frac_z_mean": st.mean(a["zero_frac"]["z_mean"] for a in S), "mean_z_sd": st.mean(a["mean"]["z_sd"] for a in S),
            "pit_max_rel_dev_pct": 100 * max(a["pit_decile_max_rel_dev"] for a in S)}
json.dump(s, open(R / "validation_extended_summary.json", "w"), indent=1)
