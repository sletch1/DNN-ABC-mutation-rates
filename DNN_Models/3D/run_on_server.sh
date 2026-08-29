#!/usr/bin/env bash
# Ship the two-stage (p1, p2, tau) generator to stat86, run it, pull the CSV back
# into DNN_Models/3D/data/.
#
# Requires key auth to stat86 (see ~/.ssh/config). If this fails with
# "Permission denied (publickey)", authorize the key once with:
#   ssh-copy-id -i ~/.ssh/id_ed25519_stat86.pub sach@stat86.stat.vt.edu
#
#   bash DNN_Models/3D/run_on_server.sh              # full run (2000 x 10)
#   bash DNN_Models/3D/run_on_server.sh 20 2         # smoke test (20 x 2)

set -euo pipefail

NDESIGN="${1:-2000}"
NREP="${2:-10}"
HOST=stat86
REMOTE=NN_ABC_3D                       # remote working dir, under $HOME
LOCAL_3D="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "$LOCAL_3D/../.." && pwd)"   # .../NN_ABC

echo "=== 1/4 shipping R code to $HOST:~/$REMOTE ==="
ssh "$HOST" "mkdir -p ~/$REMOTE/RCode ~/$REMOTE/DNN_Models/3D/data"
scp "$LOCAL_ROOT/RCode/funMBP.R" "$LOCAL_ROOT/RCode/genSlowData_3D.R" \
    "$HOST:~/$REMOTE/RCode/"

echo
echo "=== 2/4 launching generator (${NDESIGN} design points x ${NREP} reps) ==="
# nohup so the job survives SSH disconnect; poll for the completion marker.
ssh "$HOST" "cd ~/$REMOTE && rm -f run_3D.done && nohup bash -c '
  Rscript RCode/genSlowData_3D.R --ndesign $NDESIGN --nrep $NREP \
      --mut-time offspring --out slow_data_3D.csv
  Rscript RCode/genSlowData_3D.R --ndesign 200 --nrep 3 \
      --mut-time parent --out convention_check_3D.csv
  echo done > run_3D.done
' > run_3D.log 2>&1 &"

echo "launched. polling for completion..."
for _ in $(seq 1 240); do   # up to ~40 min
    if ssh "$HOST" "test -f ~/$REMOTE/run_3D.done" 2>/dev/null; then
        echo "job finished."
        break
    fi
    sleep 10
done

echo
echo "=== 3/4 server-side log (tail) ==="
ssh "$HOST" "tail -25 ~/$REMOTE/run_3D.log"

echo
echo "=== 4/4 pulling CSVs into $LOCAL_3D/data/ ==="
mkdir -p "$LOCAL_3D/data"
scp "$HOST:~/$REMOTE/DNN_Models/3D/data/slow_data_3D.csv" \
    "$HOST:~/$REMOTE/DNN_Models/3D/data/convention_check_3D.csv" \
    "$LOCAL_3D/data/"

echo
echo "=== result ==="
for f in "$LOCAL_3D/data/slow_data_3D.csv" "$LOCAL_3D/data/convention_check_3D.csv"; do
    [ -f "$f" ] && printf '%s  %s rows  %s\n' \
        "$(basename "$f")" "$(($(wc -l < "$f") - 1))" "$(du -h "$f" | cut -f1)"
done
head -1 "$LOCAL_3D/data/slow_data_3D.csv" | cut -d, -f1-11
