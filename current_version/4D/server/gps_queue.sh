#!/usr/bin/env bash
# Job queue for GPS-ABC chains with a live concurrency limit (read from $LIMIT_FILE; the server freezes above 5 MATLAB runs).
#   bash server/gps_queue.sh jobs.txt        # lines: CASE SEED N [TAG]
# A chain that has a state checkpoint resumes automatically (run_rep sets ms.resume = 1).
JOBS=$(readlink -f "$1")
cd "$(dirname "$0")/../gps_abc" || exit 1
LIMIT_FILE=${LIMIT_FILE:-$HOME/gps_limit}; [ -f "$LIMIT_FILE" ] || echo 5 > "$LIMIT_FILE"
while read -r C S N T; do
  [ -z "$C" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$(cat "$LIMIT_FILE")" ]; do sleep 30; done
  tag=${T:-}
  ( matlab -batch "run_rep($C, $S, 2, $N, '$tag')" > "log${tag}_case${C}_${S}.txt" 2>&1 < /dev/null ) &
  echo "$(date +%F_%T) started case $C seed $S N $N tag '$tag'" >> ../results/gps_queue.log
  sleep 20
done < "$JOBS"
wait
echo "$(date +%F_%T) QUEUE_DONE" >> ../results/gps_queue.log
