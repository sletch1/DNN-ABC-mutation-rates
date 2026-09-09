#!/usr/bin/env bash
#
# Runs the entire 3-D two-stage pipeline end to end: sets up a Python
# environment, trains the surrogate, runs the estimator comparison, computes
# Monte Carlo standard errors, and regenerates every figure.
#
#   ./run_all.sh             full run   (a few minutes - the reported settings)
#   ./run_all.sh --quick     smoke test (~1 min, to check it all works)
#   ./run_all.sh --with-sim  ALSO run the exact-simulator ABC baseline (hours)
#
# See HOW_TO_RUN.md for what to expect and how to run this on Windows.

set -euo pipefail
cd "$(dirname "$0")"

# The reported results were produced WITHOUT the exact-simulator ABC baseline
# (results/logs/experiment_config.json records "with_sim": false). That baseline
# re-simulates the branching process cell by cell at every MCMC iteration and
# costs hours; the 3-D study compares the two SURROGATES to each other, so it is
# off by default. --with-sim turns it back on.
QUICK=0
SIM_FLAG="--no-sim"
for arg in "$@"; do
    case "$arg" in
        --quick)    QUICK=1 ;;
        --with-sim) SIM_FLAG="" ;;
        *) echo "unknown option: $arg"; exit 1 ;;
    esac
done

if [ "$QUICK" = "1" ]; then
    REPS=2; NMCMC=300; BURNIN=100; WORKERS="--workers 2"
    echo "=== QUICK MODE: pipeline check only, NOT the reported results ==="
else
    REPS=16; NMCMC=3000; BURNIN=1000; WORKERS=""
    echo "=== FULL RUN: $REPS replicates, reported settings ==="
fi
[ -z "$SIM_FLAG" ] && echo "=== exact-simulator baseline ENABLED (expect hours) ==="

# Prefer python3, fall back to python (Windows/Git Bash usually only has `python`).
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
echo "Using interpreter: $($PY --version 2>&1)"

# ---------------------------------------------------------------------------
# Step 0: environment
# ---------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo ""
    echo "--- [0/4] Creating virtual environment and installing packages ---"
    "$PY" -m venv .venv
fi

# venv layout differs between Unix (bin/) and Windows (Scripts/).
if [ -f ".venv/bin/python" ]; then
    VENV_PY=".venv/bin/python"
else
    VENV_PY=".venv/Scripts/python.exe"
fi

"$VENV_PY" -m pip install --quiet --upgrade pip
"$VENV_PY" -m pip install --quiet -r requirements.txt
echo "Environment ready."

# ---------------------------------------------------------------------------
# Step 1: train the neural-network surrogate
# ---------------------------------------------------------------------------
echo ""
echo "--- [1/4] Training the surrogate (~30 seconds) ---"
"$VENV_PY" network/train.py --data data/slow_data_3D.csv --seed 0

# ---------------------------------------------------------------------------
# Step 2: the estimator comparison
# ---------------------------------------------------------------------------
echo ""
# Sanity checks before the comparison: that the two-stage simulator reduces to
# the constant-rate one in both limits, and that the ground truth's mutation-time
# convention is the one the pipeline assumes. Cheap, and it fails loudly rather
# than producing quietly wrong tables.
echo "--- [1b/4] Validating the simulator and the ground truth ---"
"$VENV_PY" tests/validate_simulator.py --quick

echo "--- [2/4] Running the estimator comparison: parameter recovery ---"
"$VENV_PY" abc/run_experiments.py \
    --reps "$REPS" --nmcmc "$NMCMC" --burnin "$BURNIN" --ns 4 \
    --J-grid 100 $SIM_FLAG $WORKERS

# ---------------------------------------------------------------------------
# Step 3: Monte Carlo standard errors
# ---------------------------------------------------------------------------
echo ""
echo "--- [3/4] Computing Monte Carlo standard errors ---"
"$VENV_PY" abc/mcse.py

# ---------------------------------------------------------------------------
# Step 4: figures
# ---------------------------------------------------------------------------
echo ""
echo "--- [4/4] Regenerating figures ---"
"$VENV_PY" figures/make_figures.py
"$VENV_PY" network/gen_architecture_svg.py

echo ""
echo "==========================================================="
echo "Done. Everything written to results/:"
echo "  results/tables/TABLES.md   parameter recovery, formatted for reading"
echo "  results/tables/mcse.md     which differences are real vs. noise"
echo "  results/figures/*.png      all result figures"
echo "  results/model/             the trained surrogate + its fit metrics"
echo "==========================================================="
