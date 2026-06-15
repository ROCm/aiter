#!/usr/bin/env bash
set -euo pipefail

LOG="${1:-/tmp/ar_mhc_bench_shapes.log}"
REPO="/home/yinfeliu/aiter-ar-mhc-rmsnorm"
HIP_DEV="${HIP_VISIBLE_DEVICES:-0,1}"
M_LIST="${M_LIST:-1,2,4,16,32,128,1024,2048,8192}"

SHAPES=""
IFS=',' read -ra MS <<< "$M_LIST"
for m in "${MS[@]}"; do
  SHAPES+=" ${m},4096"
done

exec > >(tee -a "$LOG") 2>&1
echo "=== AR+MHC+RMSNorm benchmark selected shapes ==="
echo "started: $(date -Is)"
echo "HIP_VISIBLE_DEVICES=$HIP_DEV"
echo "M_LIST=$M_LIST"
echo "log: $LOG"

echo "=== GPU sanity (must be idle before trusting μs) ==="
rocm-smi --showuse 2>/dev/null || true
rocm-smi --showpids 2>/dev/null || true

docker run --rm --entrypoint /bin/bash \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 32G \
  -v "$REPO:/workspace/aiter-ar-mhc-rmsnorm" \
  -w /workspace/aiter-ar-mhc-rmsnorm \
  -e PYTHONPATH=/workspace/aiter-ar-mhc-rmsnorm \
  -e HIP_VISIBLE_DEVICES="$HIP_DEV" \
  -e AITER_REBUILD=1 \
  -e AITER_USE_SYSTEM_TRITON=1 \
  vllm/vllm-openai-rocm:nightly \
  -lc "
    set -euo pipefail
    python3 -m pip install -q --no-cache-dir flydsl==0.2.0 tabulate pandas
    echo '=== smoke m=16 (rebuild) ==='
    python3 op_tests/multigpu_tests/test_fused_ar_mhc_rms.py \
      -t 2 -s 16,4096 --test split fused --force-fused
    export AITER_REBUILD=0
    echo '=== benchmark shapes (TP=2, split+fused, cudagraph default -g 1) ==='
    python3 -u op_tests/multigpu_tests/test_fused_ar_mhc_rms.py \
      -t 2 --test split fused --force-fused \
      -s${SHAPES}
  "

echo "finished: $(date -Is)"
