#!/bin/bash
# Intermittent-race stress harness. A single vmask pass is a FALSE NEGATIVE (the
# race is scheduling-dependent). This runs each candidate LOOPS times and counts
# how many runs report nondeterminism, so PASS/FAIL becomes statistical.
#
# Stage 1 (default): confirm the BASELINE races reliably enough to bisect.
#   bash stress_mtp.sh base 30
# Stage 2: once baseline races in a good fraction, bisect all candidates:
#   bash stress_mtp.sh all 30
#
# Run from /local_vol1_nobackup/qiwan/mi400_aiter on the gfx1250 box.
set -e
WHICH=${1:-base}
LOOPS=${2:-30}
CO_DIR=hsa/gfx1250/pa_decode_bf16
DEPLOYED=$CO_DIR/pa_decode_bf16_d64_page256_gqa8.co
DIAG=$CO_DIR/mtp_diag
cp -f "$DEPLOYED" "$DIAG/deployed_backup.co"

# Aggressive trigger: varlen ON (irregular work => more scheduling variance),
# batch sweep, multi-ctx + mtp 2,3 multi-config in one process.
run_stress() {
    local tag=$1 co=$2
    cp -f "$co" "$DEPLOYED"
    rm -f aiter/jit/module_pa_decode_bf16_asm.so
    AITER_REBUILD=1 python3 op_tests/test_pa_decode_bf16_asm.py --vmask \
        -m 2 3 -b 64 -kvh 8 -c 8193 16385 >/dev/null 2>&1 || true   # warm jit build once
    local races=0 fails=0 r rc
    for r in $(seq 1 $LOOPS); do
        set +e
        python3 op_tests/test_pa_decode_bf16_asm.py --vmask \
            -m 2 3 -b 17 64 -kvh 8 -c 4097 8193 16385 --varlen >/dev/null 2>&1
        rc=$?
        set -e
        [ $rc -eq 2 ] && races=$((races+1))   # INCONCLUSIVE = race
        [ $rc -eq 1 ] && fails=$((fails+1))   # bitmatch fail
    done
    printf "%-28s RACE %2d/%2d   bitfail %2d/%2d\n" "$tag" "$races" "$LOOPS" "$fails" "$LOOPS"
}

echo "== md5s =="; md5sum "$DIAG/deployed_backup.co" "$DIAG"/grp?.co "$DIAG/allsync.co
echo "== stress (LOOPS=$LOOPS, varlen, b=17,64 c=4097,8193,16385 m=2,3) =="

if [ "$WHICH" = base ]; then
    run_stress "baseline variant-54" "$DIAG/deployed_backup.co"
else
    run_stress "baseline variant-54" "$DIAG/deployed_backup.co"
    run_stress "group A (TDM)"        "$DIAG/grpA.co"
    run_stress "group B (LDS load)"   "$DIAG/grpB.co"
    run_stress "group C (gemm ds)"    "$DIAG/grpC.co"
    run_stress "group D (reduce ds)"  "$DIAG/grpD.co"
    run_stress "allsync (all)"        "$DIAG/allsync.co"
fi

cp -f "$DIAG/deployed_backup.co" "$DEPLOYED"
rm -f aiter/jit/module_pa_decode_bf16_asm.so
echo "done"
