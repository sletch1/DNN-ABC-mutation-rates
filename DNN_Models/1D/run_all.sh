#!/usr/bin/env bash
#
# Runs the entire 1-D pipeline end to end: sets up a Python environment,
# trains the surrogate, runs the full estimator comparison, computes Monte
# Carlo standard errors, and regenerates every figure.
#
#   ./run_all.sh            full run   (several hours - the paper's settings)
#   ./run_all.sh --quick    smoke test (~1-2 min, to check it all works)
#
# See HOW_TO_RUN.md for what to expect and how to run this on Windows.

set -euo pipefail
cd "$(dirname "$0")"

# --quick shrinks the expensive step so you can confirm the pipeline works
# before committing to a full run. It drops the small-p / large-J cells, which
# is where nearly all the runtime lives: the exact-simulator baseline costs
# ~340 s per 100 MCMC iterations at p=1e-4, J=100 versus ~10 s at p=1e-2, J=10.
QUICK=0
if [ "${1:-}" = "--quick" ]; then QUICK=1; fi

if [ "$QUICK" = "1" ]; then
    REPS=2; NMCMC=120; BURNIN=40
    P_GRID="1e-2"; J_GRID="10"
    # Few workers on purpose: each one fits its own GP baseline at startup
    # (cubic in design size), which with only 2 tasks would otherwise dominate.
    WORKERS="--workers 2"
    echo "=== QUICK MODE: pipeline check only, NOT paper-scale results ==="
else
    REPS=40; NMCMC=600; BURNIN=250
    P_GRID="1e-4 1e-3 1e-2"; J_GRID="10 50 100"
    WORKERS=""          # default: all cores but two
    echo "=== FULL RUN: $REPS replicates, paper settings (expect several hours) ==="
fi

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
echo "--- [1/4] Training the surrogate (fast, well under a minute) ---"
"$VENV_PY" network/train.py

# ---------------------------------------------------------------------------
# Step 2: the estimator comparison (the expensive step)
# ---------------------------------------------------------------------------
echo ""
echo "--- [2/4] Running the estimator comparison: Tables 1, 2 and 3 ---"
echo "    (this is the long one - the exact-simulator baseline is slow by design)"
"$VENV_PY" abc/run_experiments.py \
    --reps "$REPS" --nmcmc "$NMCMC" --burnin "$BURNIN" --ns 6 \
    --p-grid $P_GRID --J-grid $J_GRID $WORKERS

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
echo "  results/tables/TABLES.md   Tables 1-3, formatted for reading"
echo "  results/tables/mcse.md     which differences are real vs. noise"
echo "  results/figures/*.png      all result figures"
echo "  results/model/             the trained surrogate + its fit metrics"
echo "==========================================================="
