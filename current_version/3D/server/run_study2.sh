#!/usr/bin/env bash
# Study II inference on the observed datasets, for one model (3d or 4d): pilot step widths, DNN-ABC, NPE, SBC.
#   PY=$HOME/abc_env/bin/python bash server/run_study2.sh 4d [max_cores]
# Cores are chosen dynamically (free cores = nproc - 1-min load, at least 2), capped by max_cores.
set -euo pipefail
M=${1:?3d or 4d}; PY=${PY:-python3}; MAXC=${2:-16}
free() { local n l c; n=$(nproc); l=$(awk '{printf "%d",$1+0.5}' /proc/loadavg); c=$((n-l)); [ $c -lt 2 ] && c=2; [ $c -gt $MAXC ] && c=$MAXC; echo $c; }
D=$(echo $M | tr a-z A-Z)
$PY inference/pilot_steps.py --surrogate results/surrogate_$M.pt --obs data/obs_${M}_case1.npz,data/obs_${M}_case2.npz --out results/steps_$M.json
SC=$($PY -c "import json;print(json.load(open('results/steps_$M.json'))['chosen']['scale'])")
for c in 1 2; do
  $PY inference/dnn_abc.py --surrogate results/surrogate_$M.pt --obs data/obs_${M}_case$c.npz --out results/dnnabc_${M}_case$c.npz --step-scale $SC --cores $(free)
  $PY inference/npe.py --data data/slow_data_$D.csv --obs data/obs_${M}_case$c.npz --out results/npe_${M}_case$c.npz
done
$PY inference/sbc.py --model $M --surrogate results/surrogate_$M.pt --npe results/npe_${M}_case1.pkl --K 500 --step-scale $SC --cores $(free) --out results/sbc_$M.npz
echo STUDY2_${M}_DONE
