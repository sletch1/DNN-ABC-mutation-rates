#!/usr/bin/env bash
# DNN-ABC with the two ABC features modelled jointly (correlation head): GPS-ABC and DNN-ABC treat them as independent given theta.
# Same architecture as selected by the search; trained on all design points and on 1,500 (matched control); then DNN-ABC + calibration.
#   PY=$HOME/abc_env/bin/python bash server/run_corr.sh [arch] [cores]
set -euo pipefail
PY=${PY:-python3}; ARCH=${1:-res_128x3_gelu}; C=${2:-8}
$PY model/train_matched.py --data data/slow_data_4D.csv --arch $ARCH --n 100000 --corr --out results/surrogate_4d_rho.pt
$PY model/train_matched.py --data data/slow_data_4D.csv --arch $ARCH --n 1500 --corr --out results/surrogate_4d_rho_n1500.pt
for c in 1 2; do
  $PY inference/dnn_abc.py --surrogate results/surrogate_4d_rho.pt --obs data/obs_4d_case$c.npz --out results/dnnabcrho_4d_case$c.npz --step-scale 0.05 --cores $C
  $PY inference/dnn_abc.py --surrogate results/surrogate_4d_rho_n1500.pt --obs data/obs_4d_case$c.npz --out results/dnnabcrho1500_4d_case$c.npz --step-scale 0.05 --cores $C
done
$PY inference/sbc.py --model 4d --surrogate results/surrogate_4d_rho.pt --K 500 --step-scale 0.05 --cores $C --out results/sbc_4d_rho.npz
echo CORR_DONE
