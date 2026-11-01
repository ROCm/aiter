#!/bin/bash
set -euo pipefail
STAGE=/data/blyu/qsa/work/opt-fp8cvt-cpu-trace-20261101
mkdir -p "$STAGE"
docker run --rm --network none -v "$STAGE:/results" --entrypoint bash rocm/fw-bringup-mlperf:gfx1250-atom-20261015 -c '
 test ! -e /dev/kfd; test ! -e /dev/dri
 echo PATH=$PATH
 ls /opt /opt/venv/bin/python* /usr/local/bin/python* 2>/dev/null || true
 python3 -m pip show flydsl || true
 python3 -c "import flydsl; print(flydsl.__file__)"
 tar -czf /results/stock-flydsl-python.tar.gz -C /usr/local/lib/python3.12/dist-packages/flydsl expr _mlir/dialects
 for instruction in "v_pk_min_u16 v0, v1, v2" "v_pk_min_i16 v0, v1, v2" "v_perm_b32 v0, v1, v1, 0x01010000" "v_cvt_pk_f16_fp8 v0, v1" "v_cvt_scale_pk8_bf16_fp8 v[0:3], v[4:5], 127" "v_cvt_scale_pk8_f16_fp8 v[0:3], v[4:5], 127" "v_cvt_scale_pk8_f32_fp8 v[0:7], v[8:9], 127"; do
   echo "$instruction"
   echo "$instruction" | /opt/rocm/llvm/bin/llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding || true
 done
'
