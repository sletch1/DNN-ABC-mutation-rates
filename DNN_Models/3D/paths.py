"""Central path definitions for the 3-D two-stage surrogate package.

Every module resolves data/results locations from here, so the layout can move
without hunting down hard-coded relative paths. Import via the small sys.path
shim at the top of each runnable script (see scripts in network/, abc/, figures/).

Mirrors DNN_Models/1D/paths.py; the dataset is kept self-contained inside this
package at 3D/data/slow_data_3D.csv.
"""

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent                 # .../DNN_Models/3D
DATA = PKG_ROOT / "data" / "slow_data_3D.csv"              # two-stage ground truth

RESULTS = PKG_ROOT / "results"
FIG_DIR = RESULTS / "figures"    # all .png + architecture.svg
TABLE_DIR = RESULTS / "tables"   # TABLES.md, table1/2/3_*.csv
MODEL_DIR = RESULTS / "model"    # surrogate_3d.pt, surrogate_metrics.json
LOG_DIR = RESULTS / "logs"       # experiment_config.json, benchmark_*.md

for _d in (FIG_DIR, TABLE_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
