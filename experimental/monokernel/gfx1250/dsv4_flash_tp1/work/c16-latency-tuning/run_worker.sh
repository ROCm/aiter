#!/usr/bin/env bash
set -euo pipefail
cd /work/c16-latency-tuning
source selected.env
export EXPERIMENT=${EXPERIMENT:-rerun}
case "${1:?worker number 0..3 required}" in
 0) python native_qkv_cache.py; python native_output.py; SPLITS=4 python native_attention.py; python paired_swa.py ;;
 1) for cs in 0 1 2; do python native_compress.py "$cs"; done; RATIO=128 python paired_compressed.py ;;
 2) python native_shared.py; python native_mhc.py; python native_routing.py; python native_identity.py; RATIO=4 python paired_compressed.py ;;
 3) python native_whole_ffn.py ;;
 *) exit 2 ;;
esac
