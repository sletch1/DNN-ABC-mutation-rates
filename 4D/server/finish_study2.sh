#!/usr/bin/env bash
# After DNN-ABC for both cases: NPE for case 2 reuses the case-1 posterior (same training data; no retraining), then calibration.
#   PY=$HOME/abc_env/bin/python bash server/finish_study2.sh 4d [cores]
set -euo pipefail
M=${1:-4d}; PY=${PY:-python3}; C=${2:-8}
while [ ! -f results/dnnabc_${M}_case2.npz ]; do sleep 30; done; sleep 10
$PY inference/npe.py --data data/slow_data_4D.csv --obs data/obs_${M}_case2.npz --out results/npe_${M}_case2.npz --reuse results/npe_${M}_case1.pkl
$PY inference/sbc.py --model $M --surrogate results/surrogate_$M.pt --npe results/npe_${M}_case1.pkl --K 500 --step-scale 0.05 --cores $C --out results/sbc_$M.npz
echo FINISH_${M}_DONE
