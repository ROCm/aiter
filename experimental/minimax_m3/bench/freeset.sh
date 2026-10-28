#!/bin/bash
# user (2026-10-22, later): all 8 GPUs again "if free" -> print a free TP4 set: 1 (GPUs 4-7)
# when no job of ours holds its lock, else 0 (GPUs 0-3) when set0_idle.sh says they are idle
# (no foreign VRAM / busy time), else 1 (the caller queues behind the lock).
D=$(dirname $0)
if ! flock -n /home/vigokhal/agentx/lock_set1 true 2>/dev/null; then
  if $D/set0_idle.sh; then echo 0; exit 0; fi
fi
echo 1
