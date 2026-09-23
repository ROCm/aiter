#!/bin/bash
# N rounds, store ON/OFF alternating (order flipped on even rounds so drift
# does not land on one case). Pipe to stats.py.
set -u
N=${N:-8}; D="$(dirname "$0")"
for i in $(seq 1 "$N"); do
  if (( i % 2 )); then ORDER="0 1"; else ORDER="1 0"; fi
  for ns in $ORDER; do NO_STORE=$ns bash "$D/run_case.sh" | sed "s/^/iter=$i /"; done
done
