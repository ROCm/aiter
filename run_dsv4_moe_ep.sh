#!/usr/bin/env bash
# Sweep DSv4(-pro) MoE-EP shapes through op_tests/test_moe_ep.py
# Auto-detects the GPU arch and adapts the quant/kernel sweep:
#   - gfx950  (MI355): a8w4_mxfp4 + a4w4_mxfp4
#   - gfx1250        : a8w4_mxfp4 only (a4w4 grouped EP not yet validated)
set -u

# ---------------- config ----------------
# Resolve aiter dir: env override -> script dir (this file lives in aiter/) -> /app/aiter
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_DIR=${AITER_DIR:-$SCRIPT_DIR}
[ -f "$AITER_DIR/op_tests/test_moe_ep.py" ] || AITER_DIR=/app/aiter
PY=${PY:-python3}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
# force real a8w4 (fp8 act) for every token size
export AITER_BF16_FP8_MOE_BOUND=${AITER_BF16_FP8_MOE_BOUND:-0}
# set to 1 on the very first run so JIT kernels get built
export AITER_REBUILD=${AITER_REBUILD:-0}
# CK (composable_kernel) has no gfx1250 target -> module_moe_asm fails its
# ck_tile device static_assert. Build the CK-dependent modules CK-free so they
# fall back to ck_tile_shim.h instead of pulling ck_tile/core.hpp.
export ENABLE_CK=${ENABLE_CK:-0}

# ---------------- arch detection ----------------
# Prefer aiter's own get_gfx(); fall back to rocminfo.
GFX=${GFX:-}
if [ -z "$GFX" ]; then
  GFX=$("$PY" -c "from aiter import get_gfx; print(get_gfx())" 2>/dev/null)
fi
if [ -z "$GFX" ]; then
  GFX=$(rocminfo 2>/dev/null | grep -oiE "gfx[0-9a-f]+" | head -n1)
fi
GFX=${GFX:-unknown}
echo "detected arch: $GFX"

# quant/kernel paths to run, per arch (dsv4 = fp8fp4 -> a8w4; a4w4 optional)
case "$GFX" in
  gfx1250)
    # gfx1250 grouped EP currently validated for a8w4 only.
    QUANTS=(g1u1_a8w4_mxfp4)
    # grouped-gemm a8w4 path (set by the test too, exported here for clarity)
    export AITER_FORCE_A8W4=${AITER_FORCE_A8W4:-1}
    export AITER_USE_GROUPED_GEMM=${AITER_USE_GROUPED_GEMM:-1}
    # Running >=3 distinct token sizes in one process trips an async GPU memory
    # fault (HSA_STATUS_ERROR_EXCEPTION) in the FlyDSL grouped a8w4 kernel: each
    # token is fine on its own, but state accumulates across iterations. Isolate
    # every token in its own process until the kernel bug is fixed upstream.
    PER_TOKEN=${PER_TOKEN:-1}
    ;;
  gfx950)
    QUANTS=(g1u1_a8w4_mxfp4 g1u1_a4w4_mxfp4)
    PER_TOKEN=${PER_TOKEN:-0}
    ;;
  *)
    echo "warning: unrecognized arch '$GFX', defaulting to a8w4_mxfp4 only"
    QUANTS=(g1u1_a8w4_mxfp4)
    PER_TOKEN=${PER_TOKEN:-0}
    ;;
esac

LOGDIR=${LOGDIR:-$AITER_DIR/moe_ep_logs/${GFX}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOGDIR"

# token sweep (decode -> prefill); a8w4 boundary is 256
TOKENS=${TOKENS:-"1 16 64 128 256 512 1024 2048 4096"}

# shape table: "model_dim inter_dim E topk ep"
SHAPES=(
  "7168 3072  384 6 4"
  "7168 3072  384 6 8"
  "7168 512  384 6 8"
  "7168 768  384 6 8"
  "7168 1536 384 6 8"
  "7168 3072 48  6 8"
  # --- 385-expert / topk7 "pro" variants (ep must divide 385) ---
  # "7168 512  385 7 7"
  # "7168 768  385 7 7"
  # "7168 1536 385 7 7"
)
# ----------------------------------------

cd "$AITER_DIR" || { echo "no $AITER_DIR"; exit 1; }
SUMMARY="$LOGDIR/summary.txt"
: > "$SUMMARY"
echo "logs -> $LOGDIR"

# classify one finished run's log/rc into pass/fail/skip and record it
classify() {
  local rc="$1" log="$2" tag="$3"
  local st
  # Base the verdict on the test's own per-run PASSED/FAILED token
  # (test_moe_ep.py prints "... PASSED" / "... FAILED"). Generic words like
  # "failed"/"error" are avoided: benign noise such as "Failed to receive
  # message rc=2" would otherwise flag a passing run as failed.
  # status: skip (arch/kernel not supported) > fail > pass
  if [ "$rc" -eq 0 ] && grep -qiE "^skip |: mxfp4 requires|only .* supported" "$log"; then
    st="skip"
  elif [ "$rc" -ne 0 ] \
       || grep -qE "Traceback \(most recent call last\)" "$log" \
       || grep -qw "FAILED" "$log" \
       || ! grep -qw "PASSED" "$log"; then
    st="fail(rc=$rc)"
  else
    st="pass"
  fi
  printf "%-10s %s\n" "$st" "$tag" | tee -a "$SUMMARY"
}

for q in "${QUANTS[@]}"; do
  for s in "${SHAPES[@]}"; do
    read -r MD ID E K EP <<< "$s"
    if [ "$PER_TOKEN" = "1" ]; then
      # one process per token: avoids cross-token GPU state accumulation and
      # keeps a single token's crash from aborting the rest of the sweep.
      for tok in $TOKENS; do
        tag="${q}_md${MD}_id${ID}_E${E}_k${K}_ep${EP}_m${tok}"
        log="$LOGDIR/$tag.log"
        echo "==== RUN $tag ===="
        "$PY" -u op_tests/test_moe_ep.py \
          -t "$q" -m "$tok" -hd "$MD" -id "$ID" -e "$E" -k "$K" -ep "$EP" \
          > "$log" 2>&1
        classify $? "$log" "$tag"
        export AITER_REBUILD=0
      done
    else
      tag="${q}_md${MD}_id${ID}_E${E}_k${K}_ep${EP}"
      log="$LOGDIR/$tag.log"
      echo "==== RUN $tag ===="
      "$PY" -u op_tests/test_moe_ep.py \
        -t "$q" -m $TOKENS -hd "$MD" -id "$ID" -e "$E" -k "$K" -ep "$EP" \
        > "$log" 2>&1
      classify $? "$log" "$tag"
      export AITER_REBUILD=0
    fi
  done
done

echo
echo "===== SUMMARY ====="
cat "$SUMMARY"
echo "full logs in $LOGDIR"
