"""Redraw only the surrogate-fit figure (Fig. 1 left) from the saved checkpoint, without retraining.
    python figures/make_fit_figure.py        # writes results/figures/surrogate_fit.{png,pdf} and surrogate_parity.png
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "abc"):
    sys.path.insert(0, str(_d))
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
from train import load_surrogate, load_splits, make_plots
from paths import DATA, MODEL_DIR, FIG_DIR

surr = load_surrogate(MODEL_DIR / "surrogate_1d.pt")
make_plots(surr, load_splits(DATA, J=None, seed=0), FIG_DIR)
print("written to", FIG_DIR)
