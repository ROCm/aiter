#!/bin/bash
# Intermittent mtp=2,3 race stress harness. Single pass = false negative; this
# loops each candidate LOOPS times and counts races via the test EXIT CODE
# (2=INCONCLUSIVE/race, 1=bitmatch-fail, 0=clean).
#
# Candidates are EXPLICIT fixed-path .co files (no "current deployed" capture).
# Trigger = the ORIGINAL failing config: varlen OFF, b=64 kvh=8 c=8193,16385 m=2,3.
#
#   bash stress_mtp.sh base 50    # confirm baseline races reliably
#   bash stress_mtp.sh all  50    # then bisect all groups
#
# Run from /local_vol1_nobackup/qiwan/mi400_aiter on the gfx1250 box.
set -e
WHICH=${1:-base}
LOOPS=${2:-50}
CO_DIR=hsa/gfx1250/pa_decode_bf16
DEPLOYED=$CO_DIR/pa_decode_bf16_d64_page256_gqa8.co
DIAG=$CO_DIR/mtp_diag

run_stress() {
    local tag=$1 co=$2
    cp -f "$co" "$DEPLOYED"
    rm -f aiter/jit/module_pa_decode_bf16_asm.so
    # warm the jit .so build once (don't count this run)
    AITER_REBUILD=1 python3 op_tests/test_pa_decode_bf16_asm.py --vmask \
        -m 2 3 -b 64 -kvh 8 -c 8193 16385 >/dev/null 2>&1 || true
    local races=0 fails=0 r rc
    for r in $(seq 1 $LOOPS); do
        set +e
        python3 op_tests/test_pa_decode_bf16_asm.py --vmask \
            -m 2 3 -b 64 -kvh 8 -c 8193 16385 >/dev/null 2>&1
        rc=$?
        set -e
        [ $rc -eq 2 ] && races=$((races+1))
        [ $rc -eq 1 ] && fails=$((fails+1))
    done
    printf "%-26s [%s]  RACE %2d/%2d   bitfail %2d/%2d\n" \
        "$tag" "$(md5sum "$co" | cut -c1-8)" "$races" "$LOOPS" "$fails" "$LOOPS"
}

echo "== stress LOOPS=$LOOPS  trigger: varlen OFF, b=64 kvh=8 c=8193,16385 m=2,3 =="
if [ "$WHICH" = base ]; then
    run_stress "baseline variant-54" "$DIAG/baseline.co"
else
    run_stress "baseline variant-54" "$DIAG/baseline.co"
    run_stress "group A (TDM)"        "$DIAG/grpA.co"
    run_stress "group B (LDS load)"   "$DIAG/grpB.co"
    run_stress "group C (gemm ds)"    "$DIAG/grpC.co"
    run_stress "group D (reduce ds)"  "$DIAG/grpD.co"
    run_stress "allsync (all)"        "$DIAG/allsync.co"
fi

# leave a known state: canonical baseline deployed
cp -f "$DIAG/baseline.co" "$DEPLOYED"
rm -f aiter/jit/module_pa_decode_bf16_asm.so
echo "done (deployed restored to baseline 341aafc6)"
