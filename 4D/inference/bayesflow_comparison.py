"""Held-out comparison of the NPE implementations (sensitivity check; the paper keeps the better one, the sbi neural spline flow).
BayesFlow (Radev et al., 2020) coupling flows -- affine (depth 8, width 256) and spline (depth 6, width 128) -- are trained on exactly the data,
rows (d_bar > 0) and design-point split used for the sbi NPE, and scored by the mean held-out log q(theta | x) over unseen design points,
renormalised to the prior box so the value is comparable with sbi's. The sbi value is read from results/npe_4d_case1.npz. Needs the `bayesflow` package.
    python inference/bayesflow_comparison.py   (from 4D/; writes results/bayesflow_comparison.json)
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import bflow

R = HERE.parent / "results"
out = {"nsf": dict(heldout_logq=float(np.load(R / "npe_4d_case1.npz")["heldout_logq"]), train_s=float(np.load(R / "npe_4d_case1.npz")["train_s"]))}
for name, cfg in (("affine", dict(depth=8, transform="affine", width=256)), ("spline", dict(depth=6, transform="spline", width=128))):
    _, secs, lp = bflow.train(str(HERE.parent / "data" / "slow_data_4D.csv"), patience=50, **cfg); out[name] = dict(heldout_logq=lp, train_s=secs); print(name, out[name], flush=True)
json.dump(out, open(R / "bayesflow_comparison.json", "w"), indent=1)
