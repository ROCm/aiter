#!/usr/bin/env bash
set -euo pipefail
cd /work/c16-native-parity
mode=${1:?parts|elements|progressive|advantage|stress}; shift
export BATCH=16 ROPE_ORDERED=2 ROPE_FMA=7 POLL=2 COHERENT=0
export QA_WS=1 COMPACT_O=0 FAITHFUL_ADV=0 FAITHFUL_OB_BUFFERS=4
unset COOP DIAG
case "$mode" in
 parts) export TDM_CACHE=''; exec python faithful_parts.py "$@" ;;
 elements) export TDM_CACHE=''; exec python faithful_elements.py "$@" ;;
 progressive) export TDM_CACHE=.cg; exec python faithful_progressive.py "$@" ;;
 advantage|stress)
  export TDM_CACHE=.cg QA_WS=0 COMPACT_O=1 FAITHFUL_ADV=3 FAITHFUL_OB_BUFFERS=6
  if [[ "$mode" == stress ]]; then exec python faithful_stress.py "$@"; fi
  exec python faithful_progressive.py "$@" ;;
 *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
