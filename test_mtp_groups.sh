#!/bin/bash
# mtp=2,3 barrier-group bisect. allsync (all groups) is CLEAN; variant-54 (no
# groups) RACES. Each candidate = variant-54 + ONE re-added ALLSYNC barrier group:
#   A = TDM loads (tensor_load_to_lds)         3 barriers
#   B = K/V LDS load+ds_issue                  5 barriers
#   C = gemm_qk/gemm_pv ds_load (LDS->reg)     3 barriers
#   D = cross_wave_reduce ds_load              6 barriers
# Whichever group makes mtp 2,3 go CLEAN contains the needed barrier. If none
# alone works, the fix needs a combination (we'll test pairs next).
#
# Run from /local_vol1_nobackup/qiwan/mi400_aiter on the gfx1250 box.
set -e
CO_DIR=hsa/gfx1250/pa_decode_bf16
DEPLOYED=$CO_DIR/pa_decode_bf16_d64_page256_gqa8.co
DIAG=$CO_DIR/mtp_diag

cp -f "$DEPLOYED" "$DIAG/deployed_backup.co"

run_one() {
    local tag=$1 co=$2
    echo "############## $tag ($(md5sum "$co" | cut -d' ' -f1)) ##############"
    cp -f "$co" "$DEPLOYED"
    rm -f aiter/jit/module_pa_decode_bf16_asm.so
    AITER_REBUILD=1 python3 op_tests/test_pa_decode_bf16_asm.py --vmask \
        -m 2 3 -b 65 -kvh 8 -c 8193 16385 2>&1 | \
        grep -iE "V-mask|nondeter|PASS|FAIL|INCONC" | sed 's/^/    /'
    echo
}

run_one "baseline variant-54 (expect RACE)" "$DIAG/deployed_backup.co"
run_one "group A (TDM)"        "$DIAG/grpA.co"
run_one "group B (LDS load)"   "$DIAG/grpB.co"
run_one "group C (gemm ds)"    "$DIAG/grpC.co"
run_one "group D (reduce ds)"  "$DIAG/grpD.co"
run_one "allsync (all, expect CLEAN)" "$DIAG/allsync.co"

echo "== restoring deployed .co =="
cp -f "$DIAG/deployed_backup.co" "$DEPLOYED"
rm -f aiter/jit/module_pa_decode_bf16_asm.so
echo "done"
