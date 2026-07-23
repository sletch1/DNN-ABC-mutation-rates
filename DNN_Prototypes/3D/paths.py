"""Central path definitions for the 3-D surrogate package.

Every module resolves data/results locations from here, so the layout can move
without hunting down hard-coded relative paths. Import via the small sys.path
shim at the top of each runnable script (see scripts in network/, abc/, figures/).

Unlike the 1-D package (whose data lives in NN_ABC/data/), the 3-D dataset is
kept self-contained inside this prototype at 3D/data/slow_data_3D.csv.
"""

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent                 # .../DNN_Prototypes/3D
DATA = PKG_ROOT / "data" / "slow_data_3D.csv"              # self-contained 3-D ground truth

RESULTS = PKG_ROOT / "results"
FIG_DIR = RESULTS / "figures"    # all .png + architecture.svg
TABLE_DIR = RESULTS / "tables"   # TABLES.md, table1/2/3_*.csv
MODEL_DIR = RESULTS / "model"    # surrogate_3d.pt, surrogate_metrics.json
LOG_DIR = RESULTS / "logs"       # experiment_config.json, raw_replicates.csv, benchmark_*.md

for _d in (FIG_DIR, TABLE_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
