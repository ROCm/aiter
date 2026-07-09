#!/usr/bin/env bash
# Sweep DSv4(-pro) MoE-EP shapes through op_tests/test_moe_ep.py
# Run inside the zxe_atom container on the gfx950 (MI355) box.
set -u

# ---------------- config ----------------
AITER_DIR=/home/zxe/aiter
PY=${PY:-python3}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
# force real a8w4 (fp8 act) for every token size on gfx950
export AITER_BF16_FP8_MOE_BOUND=${AITER_BF16_FP8_MOE_BOUND:-0}
# set to 1 on the very first run so JIT kernels get built
export AITER_REBUILD=${AITER_REBUILD:-0}

LOGDIR=${LOGDIR:-$AITER_DIR/moe_ep_logs/$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOGDIR"

# quant/kernel paths to run (dsv4 = fp8fp4 -> a8w4; a4w4 optional)
QUANTS=(g1u1_a8w4_mxfp4 g1u1_a4w4_mxfp4)

# token sweep (decode -> prefill); a8w4 boundary is 256
TOKENS="1 16 64 128 256 512 1024 2048 4096"

# shape table: "model_dim inter_dim E topk ep"
SHAPES=(
  "7168 512  384 6 8"
  "7168 768  384 6 8"
  "7168 1536 384 6 8"
  "7168 3072 48  6 8"
  "4096 256  256 6 8"
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

for q in "${QUANTS[@]}"; do
  for s in "${SHAPES[@]}"; do
    read -r MD ID E K EP <<< "$s"
    tag="${q}_md${MD}_id${ID}_E${E}_k${K}_ep${EP}"
    log="$LOGDIR/$tag.log"
    echo "==== RUN $tag ===="
    "$PY" -u op_tests/test_moe_ep.py \
      -t "$q" -m $TOKENS -hd "$MD" -id "$ID" -e "$E" -k "$K" -ep "$EP" \
      > "$log" 2>&1
    rc=$?
    # crude pass/fail: python exit code + no traceback/failed-allclose in log
    if [ $rc -ne 0 ] || grep -qiE "traceback|error:|failed|mismatch" "$log"; then
      st="FAIL(rc=$rc)"
    else
      st="ok"
    fi
    printf "%-8s %s\n" "$st" "$tag" | tee -a "$SUMMARY"
    # only build once
    export AITER_REBUILD=0
  done
done

echo
echo "===== SUMMARY ====="
cat "$SUMMARY"
echo "full logs in $LOGDIR"
