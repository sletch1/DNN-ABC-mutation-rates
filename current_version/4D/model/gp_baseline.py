"""Gaussian-process baseline at several design sizes, on the same held-out designs and the same noise floor as the neural search.
The GP (anisotropic RBF + white noise, sklearn) models log10 d_bar on nonzero rows only: it cannot represent the zero mass.
    python model/gp_baseline.py --data data/slow_data_4D.csv --budgets 300,1500 --out results/gp_baseline_4D.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_search as A

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--budgets", default="300,1500"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); X, zero, y, des, idx = A.load(a.data); floor = A.noise_floor(y, zero, des, idx); out = []
    for b in map(int, a.budgets.split(",")):
        r = A.gp_baseline(X, zero, y, idx, floor, b); r = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()}
        print(b, {k: r[k] for k in ("test_x_floor", "test_cov95_raw", "train_s", "latency_us")}, flush=True); out.append(dict(budget=b, **r))
    json.dump(out, open(a.out, "w"), indent=1)
