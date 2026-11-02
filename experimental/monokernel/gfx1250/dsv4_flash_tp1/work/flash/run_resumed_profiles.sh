#!/bin/bash
set -euo pipefail
mode=${1:?arm}
for c in 1 16 128; do
 bash /work/flash/run_bench.sh "$mode" "$c" 1
done
python /work/flash/analyze_profiles.py "$mode"
