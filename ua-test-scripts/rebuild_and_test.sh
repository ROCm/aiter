#!/usr/bin/env bash
# Reliable clean rebuild + correctness test for module_unified_attention.
#
# Fixes the two recurring mistakes:
#   1. Stale .so: we hard-delete the built .so AND the build/ object dir AND the
#      copied-out .so in the jit dir, then force AITER_REBUILD=1 (which itself
#      does rm_module + clear_build). So every run compiles from scratch.
#   2. "Did it even rebuild?": we stamp the .so mtime before/after and abort if
#      the .so was not regenerated.
#
# Usage:
#   ua-test-scripts/rebuild_and_test.sh            # full clean build + test matrix
#   SKIP_BUILD=1 ua-test-scripts/rebuild_and_test.sh   # test only (use current .so)
#   MAX_JOBS=32 ua-test-scripts/rebuild_and_test.sh    # cap build parallelism
set -uo pipefail

AITER_ROOT="/root/aiter"
JIT_DIR="$AITER_ROOT/aiter/jit"
SO="$JIT_DIR/module_unified_attention.so"
TEST="$AITER_ROOT/op_tests/test_unified_attention_ck.py"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

cd "$AITER_ROOT"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "=== [build] hard-clean module_unified_attention ($(date +%H:%M:%S)) ==="
  rm -rf "$JIT_DIR/build/module_unified_attention"
  rm -f "$SO"
  before_marker="/tmp/.ua_build_before_$$"
  touch "$before_marker"
  echo "=== [build] AITER_REBUILD=1 (MAX_JOBS=${MAX_JOBS:-auto}) ==="
  # A trivial config drives the JIT build of the whole module.
  AITER_REBUILD=1 ${MAX_JOBS:+MAX_JOBS=$MAX_JOBS} python3 "$TEST" \
      -b 1 -sq 512 -sk 512 --num-heads 8,1 --head-size 128 --block-size 16 \
      --dtype bf16 --num-blocks auto --no-triton --seed 42 \
      >/tmp/ua_build_$$.log 2>&1
  if [[ ! -f "$SO" ]]; then
    echo "!!! BUILD FAILED: $SO not produced. Tail of log:"; tail -25 /tmp/ua_build_$$.log; exit 2
  fi
  if [[ ! "$SO" -nt "$before_marker" ]]; then
    echo "!!! WARNING: .so was NOT regenerated (older than build start) — possible stale build"
  fi
  echo "=== [build] OK: $(ls -la --time-style=+%H:%M:%S "$SO" | awk '{print $6,$7}') ==="
fi

echo "=== [test] regression fixtures (bug-repro guard, vs torch ref) ==="
# The canonical regression guard lives in the python suite
# (_REGRESSION_FIXTURES): non-dividing-GQA + split-KV + causal repros, etc.
# It exits non-zero if any fixture fails correctness, so just check $?.
regr_rc=0
python3 "$TEST" --swa-fixtures regression --no-triton --seed 42 \
    > /tmp/ua_regr_$$.log 2>&1 || regr_rc=$?
# Per-fixture line: "<fixture name>  <CK-vs-ref verdict>" scraped from the
# @benchmark() driver's per-call line (more robust than column-indexing the
# final markdown table).
grep -E "CK +vs ref +\| .*num_heads=" /tmp/ua_regr_$$.log | while read -r line; do
  name=$(echo "$line" | grep -oE "num_heads=\([0-9]+, [0-9]+\), head_size=[0-9]+, block_size=[0-9]+, dtype=[^,]+(, q_dtype='?[a-z0-9]+'?)?")
  verdict=$(echo "$line" | grep -oiE "passed|failed" | head -1)
  printf "  %-72s %s\n" "$name" "${verdict:-?}"
done
if [[ $regr_rc -eq 0 ]]; then
  echo "  --> regression fixtures: ALL PASS"
else
  echo "  !!! regression fixtures: FAILURE (rc=$regr_rc) — see /tmp/ua_regr_$$.log"
fi

echo "=== [test] matrix (seed 42, vs torch ref) ==="
# cfg: dtype b sq sk heads blk label
mat=(
  "bf16 2 4096 4096 12,2 64 prefill_bf16"
  "fp8  2 8192 8192 12,2 64 prefill_fp8"
  "bf16 4 1    8192 16,2 128 decode_bf16_splitkv"
  "fp8  4 1    8192 16,2 128 decode_fp8_splitkv"
  "bf16 2 1    200000 16,2 128 decode_bf16_long"
  "bf16 2 2048 2048 16,2 16 prefill_bf16_ps16"
)
pass=0; fail=0
for cfg in "${mat[@]}"; do
  set -- $cfg
  out=$(python3 "$TEST" -b $2 -sq $3 -sk $4 --num-heads $5 --head-size 128 \
        --block-size $6 --dtype $1 --num-blocks auto --no-triton \
        --seed 42 2>&1)
  verdict=$(echo "$out" | grep -iE "CK +vs ref:" | tail -1 | grep -oiE "PASS|FAIL")
  delta=$(echo "$out" | grep -iE "max abs delta" | tail -1 | sed 's/\[aiter\] *//')
  printf "  %-26s %-5s %s\n" "$7" "${verdict:-ERR}" "$delta"
  [[ "$verdict" == "PASS" ]] && pass=$((pass+1)) || fail=$((fail+1))
done
[[ $regr_rc -ne 0 ]] && fail=$((fail+1))
echo "=== [test] matrix PASS=$pass FAIL=$fail | regression rc=$regr_rc ==="
if [[ $fail -eq 0 && $regr_rc -eq 0 ]]; then
  echo "=== [test] ALL GREEN ==="; exit 0
else
  echo "=== [test] FAILURES PRESENT ==="; exit 1
fi
