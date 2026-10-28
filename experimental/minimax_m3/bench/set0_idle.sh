#!/bin/bash
# exit 0 if GPUs 0-3 are idle (no VRAM allocated, no busy time): set 0 may be used; else exit 1
for g in 0 1 2 3; do
  v=$(rocm-smi -d $g --showmemuse 2>/dev/null | grep -m1 "VRAM%" | grep -o "[0-9]*$")
  u=$(rocm-smi -d $g --showuse 2>/dev/null | grep -m1 "GPU use" | grep -o "[0-9]*$")
  [ "${v:-1}" = "0" ] && [ "${u:-1}" = "0" ] || exit 1
done
exit 0
