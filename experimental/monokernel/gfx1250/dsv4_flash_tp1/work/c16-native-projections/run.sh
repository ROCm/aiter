#!/usr/bin/env bash
set -euo pipefail
cd /work/c16-native-projections
source advantage_set.env
export EXPERIMENT="${EXPERIMENT:-reproduce}"
case "${1:-}" in
 qkv) exec python native_qkv_cache.py;;
 output) exec python native_output.py;;
 swa) exec python paired_swa.py;;
 hca) export RATIO=4; exec python paired_compressed.py;;
 csa) export RATIO=128; exec python paired_compressed.py;;
 stress-q) exec python stress.py q;;
 stress-o) exec python stress.py o;;
 *) echo 'Usage: run.sh qkv|output|swa|hca|csa|stress-q|stress-o' >&2; exit 2;;
esac
