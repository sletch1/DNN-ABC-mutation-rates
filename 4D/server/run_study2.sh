#!/usr/bin/env bash
# Study II inference on the observed datasets, for one model (3d or 4d): DNN-ABC (adaptive burn-in), NPE, SBC.
#   PY=$HOME/abc_env/bin/python bash server/run_study2.sh 4d [max_cores]
# Cores are chosen dynamically (free cores = nproc - 1-min load, at least 2), capped by max_cores.
set -euo pipefail
M=${1:?3d or 4d}; PY=${PY:-python3}; MAXC=${2:-16}
free() { local n l c; n=$(nproc); l=$(awk '{printf "%d",$1+0.5}' /proc/loadavg); c=$((n-l)); [ $c -lt 2 ] && c=2; [ $c -gt $MAXC ] && c=$MAXC; echo $c; }
D=$(echo $M | tr a-z A-Z)
SC=0.05   # initial step multiplier; widths are adapted per parameter during burn-in (see inference/dnn_abc.py)
for c in 1 2; do
  $PY inference/dnn_abc.py --surrogate results/surrogate_$M.pt --obs data/obs_${M}_case$c.npz --out results/dnnabc_${M}_case$c.npz --step-scale $SC --cores $(free)
  [ -f results/npe_${M}_case$c.npz ] || $PY inference/npe.py --data data/slow_data_$D.csv --obs data/obs_${M}_case$c.npz --out results/npe_${M}_case$c.npz
done
$PY inference/sbc.py --model $M --surrogate results/surrogate_$M.pt --npe results/npe_${M}_case1.pkl --K 500 --step-scale $SC --cores $(free) --out results/sbc_$M.npz
echo STUDY2_${M}_DONE
