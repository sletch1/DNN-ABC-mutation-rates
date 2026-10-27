#!/usr/bin/env bash
# Matched-budget control: train the chosen architecture on 1,500 design points, then run DNN-ABC on the same observed data.
#   PY=$HOME/abc_env/bin/python bash server/run_matched.sh [arch] [cores]
set -euo pipefail
PY=${PY:-python3}; ARCH=${1:-res_128x3_gelu}; C=${2:-6}
$PY model/train_matched.py --data data/slow_data_4D.csv --arch $ARCH --n 1500 --out results/surrogate_4d_n1500.pt
for c in 1 2; do
  $PY inference/dnn_abc.py --surrogate results/surrogate_4d_n1500.pt --obs data/obs_4d_case$c.npz --out results/dnnabc1500_4d_case$c.npz --step-scale 0.05 --cores $C
done
echo MATCHED_DONE
