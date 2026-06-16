#!/bin/bash
# Test the ALLSYNC barrier-removal bisect kernels one by one and log results.
#
# Usage:
#   ./run_all_bisect.sh                 # tests 00 + the 9 LIVE variants (default)
#   ./run_all_bisect.sh all             # tests all 16 (00..15; the 6 DEAD ones == 00)
#   ./run_all_bisect.sh 00 09 12        # test only the given numbers
#
# For each NN: swap in bisect_NN.co, delete the cached wrapper .so, run the
# vmask test (each input x3 -> reliable race detection), capture full output to
# a per-variant log, and append the one-line verdict to a summary log.
#
# Number -> removed-barrier map: hsa/gfx1250/pa_decode_bf16/bisect/bisect_map.txt
set -u

AITER=/local_vol1_nobackup/qiwan/mi400_aiter
CODIR="$AITER/hsa/gfx1250/pa_decode_bf16"
BISECT="$CODIR/bisect"
DST="$CODIR/pa_decode_bf16_d64_page256_gqa8.co"
CLEAN="$CODIR/pa_decode_bf16_d64_page256_gqa8.co.vmask_only.bak"   # restore target at end
LOGDIR="$AITER/bisect_logs"
mkdir -p "$LOGDIR"

# Test config that exposes the partial-page race (kv_seq_len=4097 -> OOB-wave last iter).
TEST_ARGS="-b 64 -kvh 8 -c 4097 -m 0"

# Which variants to run.
LIVE="00 02 03 04 09 10 12 13 14 15"
ALL="00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15"
if [ "$#" -eq 0 ]; then
    NUMS="$LIVE"
elif [ "$1" = "all" ]; then
    NUMS="$ALL"
else
    NUMS="$*"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOGDIR/summary_${STAMP}.log"
{
    echo "==== PA_DECODE ALLSYNC bisect run  $(date) ===="
    echo "test args: --vmask $TEST_ARGS"
    echo "variants : $NUMS"
    echo "per-variant logs: $LOGDIR/bisect_NN_${STAMP}.log"
    echo "map      : $BISECT/bisect_map.txt"
    echo "----------------------------------------------------------------"
} | tee "$SUMMARY"

for nn in $NUMS; do
    SRC="$BISECT/bisect_${nn}.co"
    if [ ! -f "$SRC" ]; then
        echo "[$nn] SKIP (no $SRC)" | tee -a "$SUMMARY"
        continue
    fi
    PER="$LOGDIR/bisect_${nn}_${STAMP}.log"
    mapline=$(grep "^${nn} " "$BISECT/bisect_map.txt" 2>/dev/null)

    cp -f "$SRC" "$DST"
    rm -f "$AITER/aiter/jit/module_pa_decode_bf16_asm.so"
    md5=$(md5sum "$DST" | cut -d' ' -f1)

    echo "===== [$nn] md5=$md5  ${mapline} =====" | tee -a "$SUMMARY"
    ( cd "$AITER" && \
      PATH=/opt/rocm-7.13-95/bin:$PATH \
      LD_LIBRARY_PATH=/opt/rocm-7.13-95/lib:$LD_LIBRARY_PATH \
      ENABLE_CK=0 ENABLE_FLYDSL=0 AITER_REBUILD=1 \
      python3 op_tests/test_pa_decode_bf16_asm.py --vmask $TEST_ARGS ) > "$PER" 2>&1
    rc=$?

    verdict=$(grep -aE "\[V-mask (PASS|FAIL|RACE)\]" "$PER" | tail -1)
    if [ -z "$verdict" ]; then
        verdict="(no verdict line; run exited rc=$rc — see $PER)"
    fi
    echo "    -> $verdict" | tee -a "$SUMMARY"
done

# Leave the kernel in the known-good (V-mask-only) state.
if [ -f "$CLEAN" ]; then
    cp -f "$CLEAN" "$DST"
    rm -f "$AITER/aiter/jit/module_pa_decode_bf16_asm.so"
    echo "----------------------------------------------------------------" | tee -a "$SUMMARY"
    echo "restored $DST <- vmask_only.bak (clean state)" | tee -a "$SUMMARY"
fi

echo "DONE. Summary: $SUMMARY" | tee -a "$SUMMARY"
