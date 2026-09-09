"""Central path definitions for the whole package.

Every module resolves data/results locations from here, so the layout can move
without hunting down hard-coded relative paths. Import via the small sys.path
shim at the top of each runnable script (see scripts in network/, abc/, figures/).

Self-contained inside this prototype at 1D/data/slow_data_1D.csv, mirroring the
3-D package's layout.

This file has no statistical content at all -- it is
pure bookkeeping (where on disk does the data live, where should output files
be written). Safe to skim past; every other script imports a few names from
here (e.g. `DATA`, `MODEL_DIR`) instead of writing out folder paths by hand.
"""

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent                 # .../DNN_Prototypes/1D
DATA = PKG_ROOT / "data" / "slow_data_1D.csv"               # self-contained 1-D ground truth

RESULTS = PKG_ROOT / "results"
FIG_DIR = RESULTS / "figures"    # all .png + architecture.svg
TABLE_DIR = RESULTS / "tables"   # TABLES.md, table1/2/3_*.csv
MODEL_DIR = RESULTS / "model"    # surrogate_1d.pt, surrogate_metrics.json
LOG_DIR = RESULTS / "logs"       # experiment_config.json, raw_replicates.csv, benchmark_*.md

for _d in (FIG_DIR, TABLE_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
