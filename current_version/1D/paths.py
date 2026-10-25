"""Central path definitions. Every script imports names from here (`DATA`,
`MODEL_DIR`, ...) instead of hard-coding folder paths."""

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
