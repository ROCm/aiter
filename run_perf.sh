#!/bin/bash
# Perf comparison: swap in each variant .co and run the NORMAL (non-vmask) test,
# which reports the kernel time (us). Compare variants to find the fastest that is
# also correct (check correctness separately with run_cross.sh).
#
# Usage:  ./run_perf.sh 99 40 50 52 53     (99 = vmask_only = no-fix lower bound)
set -u
AITER=/local_vol1_nobackup/qiwan/mi400_aiter
CODIR="$AITER/hsa/gfx1250/pa_decode_bf16"
DST="$CODIR/pa_decode_bf16_d64_page256_gqa8.co"
CTX="${CTX:-8192 4097 16385}"     # perf-representative shapes
VARS="${*:-99 40 50 52 53}"

for v in $VARS; do
    SRC="$CODIR/bisect/bisect_${v}.co"
    [ -f "$SRC" ] || { echo "[$v] no co"; continue; }
    cp -f "$SRC" "$DST"; rm -f "$AITER/aiter/jit/module_pa_decode_bf16_asm.so"
    echo "######## variant $v  md5=$(md5sum "$SRC"|cut -d' ' -f1) ########"
    ( cd "$AITER" && PATH=/opt/rocm-7.13-95/bin:$PATH \
      LD_LIBRARY_PATH=/opt/rocm-7.13-95/lib:$LD_LIBRARY_PATH \
      ENABLE_CK=0 ENABLE_FLYDSL=0 AITER_REBUILD=1 \
      python3 op_tests/test_pa_decode_bf16_asm.py -b 64 -kvh 8 -c $CTX -m 0 ) 2>&1 \
      | grep -aiE "us|max_kv|error" | grep -aviE "warning|because|cache"
done
cp -f "$CODIR/bisect/bisect_FIX.co" "$DST"; rm -f "$AITER/aiter/jit/module_pa_decode_bf16_asm.so"
echo "(restored the deployed fix)"
