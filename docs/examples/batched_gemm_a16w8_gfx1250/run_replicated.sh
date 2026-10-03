#!/bin/bash
# The replicated experiment set for ONE GPU, run strictly one process at a time.
# Launch it once per GPU, always through gpurun.sh:
#   for g in 0 1 2 3; do ./gpurun.sh $g bash ./run_replicated.sh $OUT > $OUT/rep$g.log 2>&1 & done; wait
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$HERE/wmma_peak.py" --quick
python3 "$HERE/exp_bench.py" --module "$HERE/batched_gemm_a16w8_variant.py" \
    --configs-file "$HERE/configs/replicated_dense.jsonl" --label rep_dense
python3 "$HERE/exp_bench.py" --module "$HERE/batched_gemm_a16w8_variant.py" \
    --configs-file "$HERE/configs/replicated_mla.jsonl" --strided-x --transpose-bm --label rep_mla
python3 "$HERE/wmma_peak.py" --quick
