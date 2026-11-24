#!/bin/bash
# log other users' GPU process PIDs / VRAM every 30 s (who else is on the box, and when)
while :; do
  s=$(for g in 0 1 2 3; do timeout 20 amd-smi process -g $g 2>/dev/null | grep -E "PID:|VRAM_MEM" | tr -s ' ' | tr '\n' ' '; done)
  echo "$(date +%T) ${s:-free}"
  sleep 30
done
