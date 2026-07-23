"""Round 2: is a deep ensemble of the chosen 3-D architecture worth it?

Round 1 (benchmark_arch.py) picked the single best architecture on mean-surface
fit. Deep ensembles often reduce variance and improve calibration, at k-fold
training/inference cost. Here we test whether averaging k independently seeded
copies of the chosen ResMLP beats a single network on the denoised 3-D surface --
and report seed-to-seed stability of the single model. If the ensemble's gain is
within seed noise, the cheaper single network is kept (as in the 1-D study).

Run: python benchmark_round2.py   (writes results/logs/benchmark_round2.md)
"""
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc", _ROOT / "figures"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np

from benchmark_arch import load, train_one, surface_mse
from paths import LOG_DIR

warnings.filterwarnings("ignore")

CHOSEN = dict(width=128, n_blocks=3, activation="silu", use_ln=True, dropout=0.0)
K = 5


def main():
    (X_tr, y_tr), (X_va, y_va), (X_true, y_true) = load()

    members = [train_one(CHOSEN, X_tr, y_tr, X_va, y_va, seed=s) for s in range(K)]
    singles = [surface_mse(m.predict, X_true, y_true) for m in members]

    def ens_pred(X):
        ms = np.stack([m.predict(X)[0] for m in members])
        sds = np.stack([m.predict(X)[1] for m in members])
        mbar = ms.mean(0)
        sd = np.sqrt((sds ** 2).mean(0) + ms.var(0))  # deep-ensemble predictive var
        return mbar, sd

    ens = surface_mse(ens_pred, X_true, y_true)
    smean, sstd = float(np.mean(singles)), float(np.std(singles))
    gain = (smean - ens) / smean  # fractional surface-MSE improvement of the ensemble

    lines = [
        "| model | mean-surface MSE |",
        "|---|---|",
        f"| single ResMLP (mean of {K} seeds) | {smean:.5e} +/- {sstd:.1e} |",
        f"| single ResMLP (best seed) | {min(singles):.5e} |",
        f"| deep ensemble x{K} | {ens:.5e} ({gain*100:+.1f}% vs single mean) |",
    ]
    # Cost-aware rule: an x{K} ensemble costs K-fold train + inference, so it must
    # clear a margin worth that cost (5%) -- not merely beat seed noise -- to deploy.
    COST_THRESHOLD = 0.05
    verdict = (f"ensemble improves surface fit by {gain*100:.1f}% -- above the {COST_THRESHOLD*100:.0f}% "
               f"bar for its {K}x cost, worth deploying" if gain >= COST_THRESHOLD
               else f"ensemble's {gain*100:.1f}% gain is below the {COST_THRESHOLD*100:.0f}% bar for its "
                    f"{K}x cost -> keep the single network")
    table = "\n".join(lines)
    print(table + f"\n\nverdict: {verdict}")
    (LOG_DIR / "benchmark_round2.md").write_text(
        f"# 3-D architecture benchmark, round 2: deep ensemble vs single network\n\n"
        f"{table}\n\n**Verdict:** {verdict}.\n")
    print("\nwritten -> results/logs/benchmark_round2.md")


if __name__ == "__main__":
    main()
