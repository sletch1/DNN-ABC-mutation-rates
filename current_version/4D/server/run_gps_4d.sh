#!/usr/bin/env bash
# GPS-ABC (the authors' ABC_mu2a on the paper box, patched mut_bMBP_rev for memory) on the first K observed datasets of each case.
#   bash server/run_gps_4d.sh [K=6] [N=20000] [parallel=12]
set -euo pipefail
K=${1:-6}; N=${2:-20000}; P=${3:-12}
cd gps_abc
for c in 1 2; do for s in $(seq 1 $K); do echo "$c $s"; done; done | xargs -P $P -L 1 bash -c 'matlab -batch "run_rep($0, $1, 2, '$N')" > log_case$0_$1.txt 2>&1'
echo GPS4D_DONE
