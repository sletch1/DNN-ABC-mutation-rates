#!/usr/bin/env bash
# 3-D (p1, p2, tau; delta = 1) data generation, paper settings. Run from new_version/.
#   bash server/run_3d.sh        # PY=/path/to/python overrides the interpreter
# No password is stored in this repo.
# Cores are chosen dynamically before each step: all cores minus the current 1-minute load
# (never fewer than 2), so the job takes what the shared server has free right now.
set -euo pipefail
PY=${PY:-python3}
mkdir -p data results
free_cores() {
  local n load; n=$(nproc); load=$(awk '{printf "%d", $1+0.5}' /proc/loadavg)
  local c=$(( n - load )); [ $c -lt 2 ] && c=2; echo $c
}
$PY tests/test_sim.py
C=$(free_cores); echo "training data: using $C cores"
$PY datagen/gen_training.py --model 3d --n 10000 --rep 5 --J 100 --seed 1 --cores "$C" --out data/slow_data_3D.csv
for c in 1 2; do
  C=$(free_cores); echo "observed case $c: using $C cores"
  $PY datagen/gen_observed.py --model 3d --case $c --nsim 100 --nmc 100 --cores "$C" --out data/obs_3d_case$c.npz
done
echo RUN_3D_DONE
