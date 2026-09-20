"""Round 2: focused search to try to beat the GP on mean-curve MSE.
Best single from round 1 was gelu (128,64). Try nearby configs + deep ensembles.
"""
import sys
import warnings
from pathlib import Path

# --- make sibling code folders + paths.py importable (package uses flat imports) ---
_ROOT = Path(__file__).resolve().parents[2]
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc", _ROOT / "figures"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np, pandas as pd, torch
from benchmark_arch import load, train_one, curve_mse
from surrogates import fit_gp_surrogate
from paths import LOG_DIR

warnings.filterwarnings("ignore")


def ensemble_pred(members):
    def f(x):
        ms = np.stack([m.predict(x)[0] for m in members])
        sds = np.stack([m.predict(x)[1] for m in members])
        return ms.mean(0), np.sqrt((sds ** 2).mean(0) + ms.var(0))
    return f


def main():
    (x_tr, y_tr), (x_va, y_va), (x_true, y_true) = load()
    gp = fit_gp_surrogate(x_tr, y_tr, budget=None)
    gp_mse = curve_mse(gp.predict, x_true, y_true)
    rows = [("GP (GPS-ABC baseline)", gp_mse)]

    singles = {
        "gelu (128,64)": dict(hidden_dims=(128, 64), activation="gelu", use_bn=False, dropout=0.0),
        "gelu (128,128,64)": dict(hidden_dims=(128, 128, 64), activation="gelu", use_bn=False, dropout=0.0),
        "silu (128,64)": dict(hidden_dims=(128, 64), activation="silu", use_bn=False, dropout=0.0),
        "gelu (256,128)": dict(hidden_dims=(256, 128), activation="gelu", use_bn=False, dropout=0.0),
    }
    for name, cfg in singles.items():
        # ensemble of 7
        members = [train_one(cfg, x_tr, y_tr, x_va, y_va, seed=s) for s in range(7)]
        rows.append((f"ENSEMBLE x7: {name}", curve_mse(ensemble_pred(members), x_true, y_true)))

    rows.sort(key=lambda r: r[1])
    lines = ["| model | mean-curve MSE | vs GP |", "|---|---|---|"]
    for name, mse in rows:
        rel = "—" if name.startswith("GP") else (f"{(1-mse/gp_mse)*100:+.1f}% better"
              if mse < gp_mse else f"{(mse/gp_mse-1)*100:+.1f}% worse")
        lines.append(f"| {name} | {mse:.5e} | {rel} |")
    print("\n".join(lines))
    (LOG_DIR / "benchmark_round2.md").write_text("\n".join(lines) + "\n")
    print("\nwritten -> results/benchmark_round2.md")


if __name__ == "__main__":
    main()
