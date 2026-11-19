#!/bin/bash
# Copy the original attention-path files out of the draft-base1111 checkouts for review/modification.
#   bash copy.sh
#   SRC=$HOME/draftbase_1111 DST=/dockerx/gpt_oss/opt/attention bash copy.sh
set -uo pipefail

SRC="${SRC:-$HOME/draftbase_1111}"
DST="${DST:-/dockerx/gpt_oss/opt/attention}"
SGL="$SRC/sglang"
AITER_REPO="$SRC/aiter"
# The aiter package sits one level down in the repo (aiter/aiter/ops/...); accept either layout.
if [ -d "$AITER_REPO/aiter/ops" ]; then AITER="$AITER_REPO/aiter"; else AITER="$AITER_REPO"; fi

mkdir -p "$DST/aiter" "$DST/sglang"
FAIL=0
copy() {  # copy <src> <dst>
  if [ -f "$1" ]; then cp "$1" "$2" && echo "ok    $2"
  else echo "MISS  $1"; FAIL=1; fi
}

# ---------------- aiter: files to change ----------------
# dispatcher: use_2d_kernel(), unified_attention(), _unified_attention_2d_gfx1250()
copy "$AITER/ops/triton/attention/unified_attention.py"                             "$DST/aiter/unified_attention_origin.py"
# current gfx1250 prefill kernel (_unified_attention_gluon_kernel_2d)
copy "$AITER/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py"   "$DST/aiter/unified_attention_2d_gfx1250_origin.py"
# config lookup: get_unified_attention_config()
copy "$AITER/ops/triton/utils/unified_attention_utils.py"                           "$DST/aiter/unified_attention_utils_origin.py"
# gfx1250 gluon tile configs
copy "$AITER/ops/triton/configs/gfx1250/gluon/attention/unified_attention/DEFAULT.json" "$DST/aiter/unified_attention_gfx1250_gluon_DEFAULT_origin.json"
# accuracy test and benchmark
copy "$AITER_REPO/op_tests/triton_tests/attention/test_unified_attention.py"        "$DST/aiter/test_unified_attention_origin.py"
copy "$AITER_REPO/op_tests/op_benchmarks/triton/bench_unified_attention.py"         "$DST/aiter/bench_unified_attention_origin.py"

# ---------------- aiter: references (read only) ----------------
copy "$AITER/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py"     "$DST/aiter/unified_attention_3d_gfx1250_origin.py"
copy "$AITER/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_reduce.py" "$DST/aiter/unified_attention_reduce_gfx1250_origin.py"
copy "$AITER/ops/triton/_gluon_kernels/gfx1250/attention/mha_fp8.py"                  "$DST/aiter/mha_fp8_gfx1250_origin.py"
copy "$AITER/ops/triton/_gluon_kernels/gfx1250/attention/mha_mxfp8.py"                "$DST/aiter/mha_mxfp8_gfx1250_origin.py"
copy "$AITER/ops/triton/_gluon_kernels/gfx1250/attention/pa_prefill_sparse.py"        "$DST/aiter/pa_prefill_sparse_gfx1250_origin.py"
copy "$AITER/ops/triton/_gluon_kernels/common/utils.py"                               "$DST/aiter/gluon_common_utils_origin.py"

# ---------------- sglang: files to change ----------------
# _forward_extend_unified() -> unified_attention(...) for gpt-oss prefill
copy "$SGL/python/sglang/srt/layers/attention/aiter_backend.py"   "$DST/sglang/aiter_backend_origin.py"
# register a new SGLANG_* switch
copy "$SGL/python/sglang/srt/environ.py"                          "$DST/sglang/environ_origin.py"

# ---------------- sglang: references (read only) ----------------
# forward_extend_vectorized_5d(): prefill path for the shuffled (5D) KV cache
copy "$SGL/python/sglang/srt/layers/attention/aiter_utils.py"     "$DST/sglang/aiter_utils_origin.py"
copy "$SGL/python/sglang/srt/models/gpt_oss.py"                   "$DST/sglang/gpt_oss_origin.py"
copy "$SGL/python/sglang/srt/layers/radix_attention.py"           "$DST/sglang/radix_attention_origin.py"

# record the exact source revisions
{
  echo "copied $(date '+%Y-%m-%d %H:%M:%S') from $SRC"
  for r in sglang aiter triton; do
    [ -d "$SRC/$r/.git" ] && echo "$r $(git -C "$SRC/$r" rev-parse --abbrev-ref HEAD) $(git -C "$SRC/$r" rev-parse --short HEAD)"
  done
} > "$DST/SOURCE_REVISIONS.txt"
echo "revisions -> $DST/SOURCE_REVISIONS.txt"

[ "$FAIL" = 0 ] && echo "done" || { echo "some files are missing (see MISS lines)"; exit 1; }
