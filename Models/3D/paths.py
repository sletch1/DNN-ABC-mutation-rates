"""Central path definitions. Every script imports names from here (`DATA`,
`MODEL_DIR`, ...) instead of hard-coding folder paths. Mirrors Models/1D/paths.py."""

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent                 # .../Models/3D
DATA = PKG_ROOT / "data" / "slow_data_3D.csv"              # two-stage ground truth

RESULTS = PKG_ROOT / "results"
FIG_DIR = RESULTS / "figures"    # all .png + architecture.svg
TABLE_DIR = RESULTS / "tables"   # TABLES.md, table1/2/3_*.csv
MODEL_DIR = RESULTS / "model"    # surrogate_3d.pt, surrogate_metrics.json
LOG_DIR = RESULTS / "logs"       # experiment_config.json, benchmark_*.md

for _d in (FIG_DIR, TABLE_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
