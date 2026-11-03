#!/bin/bash
# Round DRAFT: CPU-only (no --device, --network none) compile + ISA + gates.
# Usage: bash host_compile_draft_cpu.sh /abs/new-source.tar.gz /abs/base-source.tar.gz
set -euo pipefail
NEW=$(realpath "${1:?new source archive required}")
BASE=$(realpath "${2:?base source archive required}")
STAGE=$(dirname "$NEW")
IMG=rocm/fw-bringup-mlperf:gfx1250-atom-20261015
SRC_NEW=$(mktemp -d "$STAGE/source-draft.XXXXXX")
SRC_BASE=$(mktemp -d "$STAGE/source-base.XXXXXX")
RESULTS=$(mktemp -d "$STAGE/results-draft.XXXXXX")
tar -xzf "$NEW" -C "$SRC_NEW"; tar -xzf "$BASE" -C "$SRC_BASE"
mkdir -p "$RESULTS/new" "$RESULTS/base"; chmod -R a+rwx "$RESULTS"
printf 'SRC_NEW=%s\nSRC_BASE=%s\nRESULTS=%s\n' "$SRC_NEW" "$SRC_BASE" "$RESULTS"
dock() { docker run --rm --network none -v "$1:/source:ro" -v "$RESULTS:/results" -w /source \
  -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
  -e QSA_GOLDEN_DIR=/source/golden_baseline --entrypoint bash "$IMG" -c "$2"; }
PRE='set -euo pipefail; test ! -e /dev/kfd; test ! -e /dev/dri; G2=aiter/ops/flydsl/kernels/mqa_logits/_g2;
python3 -B -c "import importlib.metadata as m; assert m.version(\"flydsl\")==\"0.3.3.dev886\""'
# 1. default-off byte identity: stock compile_only.py on base and new (default env)
for side in base new; do
  src=$SRC_NEW; [ $side = base ] && src=$SRC_BASE
  dock "$src" "$PRE
    export FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 QSA_RESULTS=/results/$side/default
    export FLYDSL_DUMP_DIR=/results/$side/default/isa FLYDSL_RUNTIME_CACHE_DIR=/results/$side/default/cache
    mkdir -p \$QSA_RESULTS; python3 -B \$G2/compile_only.py > \$QSA_RESULTS/compile.log 2>&1 || { tail -40 \$QSA_RESULTS/compile.log; exit 1; }
    grep '^\[COMPILED\]\|^\[FAILED\]' \$QSA_RESULTS/compile.log"
done
# 2. every draft variant + QSA_R16_NOBAR=1 r16 TDM kernel
dock "$SRC_NEW" "$PRE
  export QSA_R16_NOBAR=1 FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 QSA_RESULTS=/results/new/draft
  export FLYDSL_DUMP_DIR=/results/new/draft/isa FLYDSL_RUNTIME_CACHE_DIR=/results/new/draft/cache
  mkdir -p \$QSA_RESULTS; python3 -B \$G2/compile_draft_cpu.py > \$QSA_RESULTS/compile.log 2>&1 || { tail -80 \$QSA_RESULTS/compile.log; exit 1; }
  grep '^\[COMPILED\]\|^\[FAILED\]' \$QSA_RESULTS/compile.log"
# 3. CPU gates (existing + new)
dock "$SRC_NEW" "set -uo pipefail; G2=aiter/ops/flydsl/kernels/mqa_logits/_g2; rc=0
  run() { local n=\$1; shift; \"\$@\" > /results/gate-\$n.log 2>&1; local s=\$?; echo \"\$n exit=\$s\"; tail -3 /results/gate-\$n.log; [ \$s -eq 0 ] || rc=1; }
  run draft python3 -B \$G2/test_draft_cpu.py -v
  QSA_BF16_K_LOAD=raw run contract python3 -B \$G2/test_contract_cpu.py
  QSA_BF16_K_LOAD=raw run address python3 -B \$G2/audit_addressing_cpu.py
  QSA_BF16_K_LOAD=raw run q-lds python3 -B \$G2/test_q_lds_cpu.py
  QSA_BF16_K_LOAD=raw QSA_TRACE_ROWS_PER_BLOCK=16 run trace16 python3 -B \$G2/trace_check_cpu.py
  QSA_BF16_K_LOAD=raw QSA_TRACE_ROWS_PER_BLOCK=32 run trace32 python3 -B \$G2/trace_check_cpu.py
  run k-tdm python3 -B \$G2/test_k_tdm_cpu.py -v
  run routing python3 -B \$G2/test_bf16_routing_cpu.py -v
  run fp8-k-tdm python3 -B \$G2/test_fp8_k_tdm_cpu.py -v
  run fp8-cvt python3 -B \$G2/test_fp8_cvt_cpu.py -v
  run bm64 python3 -B \$G2/test_bf16_bm64_cpu.py -v
  run persist python3 -B \$G2/test_persist_cpu.py -v
  run nosync python3 -B \$G2/test_nosync_i32_cpu.py -v
  run pad python3 -B \$G2/test_pad_bucket_cpu.py -v
  run launch python3 -B \$G2/test_launch_cpu.py -v
  QSA_DRAFT_PATH=1 QSA_R16_NOBAR=1 run k-tdm-optin python3 -B \$G2/test_k_tdm_cpu.py -v
  QSA_DRAFT_PATH=1 QSA_R16_NOBAR=1 QSA_BF16_K_LOAD=raw run contract-optin python3 -B \$G2/test_contract_cpu.py
  echo ALL_GATES_RC=\$rc"
# 4. ISA identity + draft ISA stats (host side)
cd "$RESULTS"
echo '== default-off ISA identity (base vs new) =='
for f in $(cd base/default/isa && ls -d */ | tr -d /); do
  a=base/default/isa/$f/21_final_isa.s; b=new/default/isa/$f/21_final_isa.s
  if cmp -s "$a" "$b"; then echo "IDENTICAL $f $(sha256sum < "$a" | cut -c1-16)"; else echo "DIFFERENT $f"; fi
done
echo '== draft ISA stats =='
for d in new/draft/isa/*draft*/ new/draft/isa/*nobar*/ base/default/isa/*r16_c32_flydsl_tdm/ base/default/isa/*fp8_wmma*r16_c64_flydsl/; do
  s=$d/21_final_isa.s; n=$(basename "$d")
  printf '%s vgpr=%s sgpr=%s lds=%s scratch=%s instr=%s wmma=%s vmem_load=%s tdm=%s s_wait_tensor=%s barrier=%s cvt_pk8=%s valu_v=%s ds_load=%s\n' "$n" \
    "$(grep -m1 -E '\.vgpr_count:' "$s" | awk '{print $2}')" \
    "$(grep -m1 -E '\.sgpr_count:' "$s" | awk '{print $2}')" \
    "$(grep -m1 -E '\.group_segment_fixed_size:' "$s" | awk '{print $2}')" \
    "$(grep -m1 -E '\.private_segment_fixed_size:' "$s" | awk '{print $2}')" \
    "$(grep -cE '^\s+[sv]_|^\s+buffer_|^\s+ds_|^\s+tensor_|^\s+global_' "$s")" \
    "$(grep -c 'v_wmma' "$s")" "$(grep -cE 'buffer_load' "$s")" "$(grep -cE 'tensor_load' "$s")" \
    "$(grep -c 's_wait_tensorcnt' "$s")" "$(grep -c 's_barrier' "$s")" "$(grep -c 'v_cvt_scale_pk8' "$s")" \
    "$(grep -cE '^\s+v_' "$s")" "$(grep -cE '^\s+ds_load' "$s")"
done
sha256sum "$SRC_NEW"/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
echo "RESULTS=$RESULTS"
