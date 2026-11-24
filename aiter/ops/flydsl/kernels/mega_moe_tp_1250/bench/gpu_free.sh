#!/bin/bash
# gpu_free.sh [gpus...] : exit 0 if every listed GPU (default 0-3) has no compute process
# and < 5% busy; prints the busy ones. Waits up to $WAIT seconds (default 0) for them.
gpus=${@:-0 1 2 3}
deadline=$(( $(date +%s) + ${WAIT:-0} ))
while :; do
  busy=""
  for g in $gpus; do
    use=$(timeout 20 rocm-smi -d $g --showuse 2>/dev/null | grep -oE "GPU use \(%\): *[0-9]+" | grep -oE "[0-9]+$" || echo 999)
    procs=$(timeout 20 amd-smi process -g $g 2>/dev/null | grep -c "PID:" )
    if [ "${use:-0}" -ge 5 ] || [ "${procs:-0}" -gt 0 ]; then busy="$busy $g(use=${use},procs=${procs})"; fi
  done
  [ -z "$busy" ] && { echo "GPUs free: $gpus"; exit 0; }
  [ $(date +%s) -ge $deadline ] && { echo "BUSY:$busy"; exit 1; }
  sleep 10
done
