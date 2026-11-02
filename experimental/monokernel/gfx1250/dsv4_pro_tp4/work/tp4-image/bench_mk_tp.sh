#!/bin/bash
set -euo pipefail
cd /work
for c in "${@:-1 8 16 32 64 128}"; do
 for bucket in $c; do
  bash /opt/dsv4-mk-tp/reproduction/dsv4-monokernel/run_bench.sh mk_tp_run1 "$bucket" 0
  bash /opt/dsv4-mk-tp/reproduction/dsv4-monokernel/run_bench.sh mk_tp_run2 "$bucket" 0
 done
done
