#!/bin/bash
# No-device stock compiler: prove default == fast and current stays legacy.
set -euo pipefail
SOURCE=$(realpath "${1:?source}")
RESULTS=$(realpath -m "${2:?results}")
mkdir -p "$RESULTS"
docker run --rm --network none -v "$SOURCE:/source:ro" -v "$RESULTS:/results" -w /source \
 -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
 -e FLYDSL_DUMP_IR=1 -e FLYDSL_DEBUG_DUMP_ASM=1 \
 --entrypoint bash rocm/fw-bringup-mlperf:gfx1250-atom-20261015 -c '
 set -euo pipefail
 test ! -e /dev/kfd; test ! -e /dev/dri
 python3 -B -c "import importlib.metadata as m; assert m.version(\"flydsl\")==\"0.3.3.dev886\""
 G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
 for mode in default current fast; do
   unset QSA_FP8_CVT
   if [ "$mode" != default ]; then export QSA_FP8_CVT=$mode; fi
   export QSA_RESULTS=/results/$mode FLYDSL_DUMP_DIR=/results/$mode/isa FLYDSL_RUNTIME_CACHE_DIR=/results/$mode/cache
   mkdir -p "$QSA_RESULTS"
   python3 -B "$G2/compile_fp8cvt_only_cpu.py" > "$QSA_RESULTS/compile.log" 2>&1 || { tail -80 "$QSA_RESULTS/compile.log"; exit 1; }
   tail -5 "$QSA_RESULTS/compile.log"
   python3 -B "$G2/compile_fp8cvt_micro_cpu.py" > "$QSA_RESULTS/micro-compile.log" 2>&1 || { tail -80 "$QSA_RESULTS/micro-compile.log"; exit 1; }
   tail -2 "$QSA_RESULTS/micro-compile.log"
 done
 python3 -B "$G2/test_fp8_cvt_cpu.py" -v > /results/fp8-cvt.log 2>&1
 tail -5 /results/fp8-cvt.log
 sha256sum /source/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py > /results/source.sha256
'
