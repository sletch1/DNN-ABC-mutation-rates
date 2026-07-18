"""Central path definitions for the whole package.

Every module resolves data/results locations from here, so the layout can move
without hunting down hard-coded relative paths. Import via the small sys.path
shim at the top of each runnable script (see scripts in network/, abc/, figures/).
"""

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent                 # .../mutation_rate_1d_surrogate
DATA = PKG_ROOT.parent.parent / "data" / "slow_data_1D.csv"  # NN_ABC/data/slow_data_1D.csv

RESULTS = PKG_ROOT / "results"
FIG_DIR = RESULTS / "figures"    # all .png + architecture.svg
TABLE_DIR = RESULTS / "tables"   # TABLES.md, table1/2/3_*.csv
MODEL_DIR = RESULTS / "model"    # surrogate_1d.pt, surrogate_metrics.json
LOG_DIR = RESULTS / "logs"       # experiment_config.json, raw_replicates.csv, benchmark_*.md

for _d in (FIG_DIR, TABLE_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
