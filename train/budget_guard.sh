#!/bin/zsh
# Budget guard for paid Lium training. Every 45 min: log balance and pull the pod's
# /root/ckpt_sync (latest resumable state: adapters, optimizer, step, data cursor) to <repo>/ckpt/pod/.
# Below FLOOR dollars: final pull, then terminate the pod so a run can never drain the account.
# Usage: nohup train/budget_guard.sh <pod> [floor] > budget_guard.log 2>&1 &   Stop: touch <repo>/STOP_GUARD
POD=$1; FLOOR=${2:-0.75}
REPO=$(cd "$(dirname "$0")/.." && pwd); OUT=$REPO/ckpt/pod; mkdir -p "$OUT"
cd $HOME  # lium reads ~/.lium config
pull() {
  lium exec "$POD" 'cd /root && [ -d ckpt_sync ] && tar czf ckpt_sync.tgz.part ckpt_sync && mv ckpt_sync.tgz.part ckpt_sync.tgz' >/dev/null 2>&1 &&
    lium scp "$POD" /root/ckpt_sync.tgz "$OUT/" -d >/dev/null 2>&1 &&
    echo "$(date +%H:%M) pulled $(du -h "$OUT/ckpt_sync.tgz" | cut -f1)"
}
while [[ ! -e $REPO/STOP_GUARD ]]; do
  bal=$(lium balance 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
  echo "$(date +%H:%M) balance \$${bal:-?}"
  if [[ -n $bal ]] && (( bal < FLOOR )); then
    echo "below \$$FLOOR: final pull + terminate $POD"; pull; lium rm "$POD" --yes; exit 0
  fi
  pull
  sleep 2700
done
