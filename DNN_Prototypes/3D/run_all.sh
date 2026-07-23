#!/usr/bin/env bash
# Autonomous end-to-end build & test of the 3-D DNN-ABC surrogate on stat86.
# Runs every stage, records pass/fail + timing, continues past failures, packages
# the results, and emails a plaintext summary. Designed to be launched with nohup
# so it survives SSH disconnect / laptop shutdown.
#
#   nohup bash run_all.sh > results/logs/run_all.log 2>&1 &

set -u
cd "$(dirname "$0")"
DIR="$(pwd)"
PY="$HOME/miniconda3/envs/abc/bin/python"
STATUS="$DIR/results/logs/pipeline_status.txt"
mkdir -p results/logs results/figures results/tables results/model

# one BLAS/OMP thread per process so the 30-way multiprocessing pool in
# run_experiments does not oversubscribe the 32 cores.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

: > "$STATUS"
echo "=== run_all.sh start $(date) ==="
echo "python: $PY"; "$PY" -c "import torch,sklearn,pandas;print('env ok', torch.__version__)" || {
  echo "ENV BROKEN - aborting"; exit 1; }

run_step () {  # name  cmd...
  local name="$1"; shift
  echo; echo ">>> [$name] $(date '+%H:%M:%S')"; local t0=$(date +%s)
  "$@"; local code=$?
  local t1=$(date +%s)
  echo "STEP|$name|$code|$((t1 - t0))" >> "$STATUS"
  echo "<<< [$name] exit=$code  $(( (t1 - t0) / 60 ))m$(( (t1 - t0) % 60 ))s"
  return 0
}

run_step validate_simulator   "$PY" tests/validate_simulator.py
run_step train_surrogate      "$PY" network/train.py
run_step architecture_search  "$PY" network/architecture_search/benchmark_arch.py
run_step ensemble_check        "$PY" network/architecture_search/benchmark_round2.py
run_step surrogate_quality    "$PY" tests/surrogate_quality.py
run_step gp_scaling           "$PY" tests/gp_scaling.py
run_step abc_tables           "$PY" abc/run_experiments.py --reps 32 --nmcmc 600 \
                                     --burnin 250 --ns 6 --workers 30 --timing-iters 40
run_step abc_coverage         "$PY" tests/abc_coverage.py
run_step figures              "$PY" figures/make_figures.py
run_step architecture_svg     "$PY" network/gen_architecture_svg.py

# package everything for pulling
echo; echo ">>> packaging results"
tar -czf "$HOME/3D_results.tgz" -C "$DIR" results 2>/dev/null && \
  echo "wrote $HOME/3D_results.tgz ($(du -h "$HOME/3D_results.tgz" | cut -f1))"

# completion marker (a future session polls for this)
date > results/logs/PIPELINE_DONE

# email the summary; on any failure, send a minimal fallback so the user still hears back
echo; echo ">>> emailing report"
if ! "$PY" report.py; then
  "$PY" send_report.py --subject "[3D DNN-ABC] run finished (report builder failed)" \
        --body "The 3-D pipeline finished on stat86 but report.py errored. Pipeline status:
$(cat "$STATUS" 2>/dev/null)

Pull ~/3D_results.tgz and inspect results/logs/ on the server." || echo "EMAIL FAILED"
fi

echo "=== run_all.sh done $(date) ==="
