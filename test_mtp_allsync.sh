#!/bin/bash
# Decisive mtp=2,3 race diagnostic: swap in the ALLSYNC (maximal-barrier) .co and
# re-run the V-mask determinism test for mtp 2 3.
#
#   allsync CLEAN for mtp 2,3  -> race is a barrier variant-54 removed (the bisect
#                                 that approved the removal was mtp=0-only and could
#                                 not see the m=1 dependency). Next: bisect which one.
#   allsync STILL RACES        -> NOT a barrier; logic bug in the m=1 (second m-tile)
#                                 split path. Next: chase the logic.
#
# Run from /local_vol1_nobackup/qiwan/mi400_aiter on the gfx1250 box.
set -e
CO_DIR=hsa/gfx1250/pa_decode_bf16
DEPLOYED=$CO_DIR/pa_decode_bf16_d64_page256_gqa8.co
ALLSYNC=$CO_DIR/mtp_diag/allsync.co

echo "== backing up currently-deployed .co =="
cp -f "$DEPLOYED" "$CO_DIR/mtp_diag/deployed_backup.co"
md5sum "$DEPLOYED" "$ALLSYNC"

echo "== installing allsync .co + clearing jit cache =="
cp -f "$ALLSYNC" "$DEPLOYED"
rm -f aiter/jit/module_pa_decode_bf16_asm.so

echo "== running vmask determinism test (mtp 2 3) =="
AITER_REBUILD=1 python3 op_tests/test_pa_decode_bf16_asm.py --vmask \
    -m 2 3 -b 65 -kvh 8 -c 8193 16385 2>&1 | tee /tmp/mtp_allsync.log | \
    grep -iE "V-mask|det_zero|det_nan|nondeter|PASS|FAIL|INCONC|mtp=|bitmatch"

echo
echo "== restoring deployed .co =="
cp -f "$CO_DIR/mtp_diag/deployed_backup.co" "$DEPLOYED"
rm -f aiter/jit/module_pa_decode_bf16_asm.so
echo "done (full log: /tmp/mtp_allsync.log)"
