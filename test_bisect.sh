#!/bin/bash
# Bisect which ALLSYNC barrier is essential for the partial-page race.
# Usage:  ./test_bisect.sh NN            (NN = 00..15)
#         ./test_bisect.sh NN -c 4097    (override test args)
#
# NN=00 -> FULL allsync (all 15 barriers) = clean reference.
# NN=KK -> barrier KK REMOVED (other 14 present). If the race reappears
#          (det_zero>0 / det_nan>0 / nondeterministic=True), barrier KK is the
#          one that matters -> tell me "KK" and I'll point to the exact load.
#
# Each run: swap in bisect_NN.co, delete the cached wrapper .so, rebuild+run.
set -u
NN="$1"; shift || true
AITER=/local_vol1_nobackup/qiwan/mi400_aiter
CODIR="$AITER/hsa/gfx1250/pa_decode_bf16"
SRC="$CODIR/bisect/bisect_${NN}.co"
DST="$CODIR/pa_decode_bf16_d64_page256_gqa8.co"

if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found. Available:"; ls "$CODIR/bisect/" 2>/dev/null; exit 1
fi

echo "=== bisect $NN : $(grep "^${NN} " "$CODIR/bisect/bisect_map.txt") ==="
cp -f "$SRC" "$DST"
echo "co md5: $(md5sum "$DST" | cut -d' ' -f1)"
# Force the JIT wrapper to rebuild and reload the (swapped) .co.
rm -f "$AITER/aiter/jit/module_pa_decode_bf16_asm.so"

# Default test args: the partial-page config that exposes the race.
ARGS="${*:- -b 64 -kvh 8 -c 4097 -m 0}"
cd "$AITER"
PATH=/opt/rocm-7.13-95/bin:$PATH \
LD_LIBRARY_PATH=/opt/rocm-7.13-95/lib:$LD_LIBRARY_PATH \
ENABLE_CK=0 ENABLE_FLYDSL=0 AITER_REBUILD=1 \
python3 op_tests/test_pa_decode_bf16_asm.py --vmask $ARGS
