#!/bin/bash
# Cross-config robustness test: for each variant, run the vmask test across ALL
# the configs that have exposed the race, in ONE python invocation (so you get
# one [V-mask] line PER ctx). A variant is robust ONLY if it PASSes every ctx.
#
# Usage:  ./run_cross.sh 40 44 25 22      (variant numbers; default below)
set -u
AITER=/local_vol1_nobackup/qiwan/mi400_aiter
CODIR="$AITER/hsa/gfx1250/pa_decode_bf16"
DST="$CODIR/pa_decode_bf16_d64_page256_gqa8.co"
CTX="8192 8193 4097 16385"          # full(8192,pad=0) + partial(8193/4097/16385)
VARS="${*:-40 44 25 22}"
LOG="$AITER/bisect_logs/cross_$(date +%H%M%S).log"
mkdir -p "$AITER/bisect_logs"

for v in $VARS; do
    SRC="$CODIR/bisect/bisect_${v}.co"
    [ -f "$SRC" ] || { echo "no bisect_${v}.co" | tee -a "$LOG"; continue; }
    echo "############## variant $v  md5=$(md5sum "$SRC"|cut -d' ' -f1) ##############" | tee -a "$LOG"
    cp -f "$SRC" "$DST"
    rm -f "$AITER/aiter/jit/module_pa_decode_bf16_asm.so"
    ( cd "$AITER" && PATH=/opt/rocm-7.13-95/bin:$PATH \
      LD_LIBRARY_PATH=/opt/rocm-7.13-95/lib:$LD_LIBRARY_PATH \
      ENABLE_CK=0 ENABLE_FLYDSL=0 AITER_REBUILD=1 \
      python3 op_tests/test_pa_decode_bf16_asm.py --vmask -b 64 -kvh 8 -c $CTX -m 0 ) 2>&1 \
      | grep -aE "\[V-mask" | tee -a "$LOG"
done
# restore robust allsync
cp -f "$CODIR/bisect/bisect_00.co" "$DST"; rm -f "$AITER/aiter/jit/module_pa_decode_bf16_asm.so"
echo "(restored allsync). log: $LOG" | tee -a "$LOG"
