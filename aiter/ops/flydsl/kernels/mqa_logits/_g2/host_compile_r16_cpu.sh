#!/bin/bash
# Round R16: CPU-only stock-FlyDSL compile + gates. No GPU devices are exposed.
# Usage: bash host_compile_r16_cpu.sh /abs/new-source.tar.gz /abs/base-source.tar.gz
# Each archive holds aiter/ops/flydsl/kernels/mqa_logits (kernel + _g2) and the
# two public golden_baseline/qsa_{golden,shapes}.py modules (no __pycache__).
set -euo pipefail
NEW=$(realpath "${1:?new source archive required}")
BASE=$(realpath "${2:?base source archive required}")
STAGE=$(dirname "$NEW")
IMG=rocm/fw-bringup-mlperf:gfx1250-atom-20261015
SRC_NEW=$(mktemp -d "$STAGE/source-r16.XXXXXX")
SRC_BASE=$(mktemp -d "$STAGE/source-base.XXXXXX")
RESULTS=$(mktemp -d "$STAGE/results-r16.XXXXXX")
tar -xzf "$NEW" -C "$SRC_NEW"
tar -xzf "$BASE" -C "$SRC_BASE"
mkdir -p "$RESULTS/new" "$RESULTS/base"
printf 'SRC_NEW=%s\nSRC_BASE=%s\nRESULTS=%s\n' "$SRC_NEW" "$SRC_BASE" "$RESULTS"
compile() {  # $1 source dir, $2 results subdir, $3 modes
  docker run --rm --network none -v "$1:/source:ro" -v "$RESULTS/$2:/results" -w /source \
    -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
    -e FLYDSL_DUMP_IR=1 -e FLYDSL_DEBUG_DUMP_ASM=1 -e MODES="$3" \
    --entrypoint bash "$IMG" -c '
      set -euo pipefail
      ls /dev/kfd /dev/dri 2>/dev/null && { echo "GPU device visible: abort"; exit 9; } || true
      G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
      python3 -B -c "import importlib.metadata as m; assert m.version(\"flydsl\")==\"0.3.3.dev886\"; print(\"FlyDSL\", m.version(\"flydsl\"))"
      for mode in $MODES; do
        if [ "$mode" = tdm64 ]; then export QSA_BF16_K_LOAD=tdm QSA_BF16_R16_TDM_COLS=64;
        else export QSA_BF16_K_LOAD=$mode; unset QSA_BF16_R16_TDM_COLS; fi
        export QSA_RESULTS=/results/$mode FLYDSL_DUMP_DIR=/results/$mode/isa FLYDSL_RUNTIME_CACHE_DIR=/results/$mode/cache
        mkdir -p "$QSA_RESULTS"
        python3 -B "$G2/compile_only.py" > "$QSA_RESULTS/compile.log" 2>&1 || { tail -60 "$QSA_RESULTS/compile.log"; exit 1; }
        grep "^\[COMPILED\]\|^\[FAILED\]" "$QSA_RESULTS/compile.log"
      done'
}
compile "$SRC_BASE" base "auto raw tdm"
compile "$SRC_NEW" new "auto raw tdm tdm64"
docker run --rm --network none -v "$SRC_NEW:/source:ro" -v "$RESULTS:/results" -w /source \
  -e PYTHONDONTWRITEBYTECODE=1 -e QSA_GOLDEN_DIR=/source/golden_baseline \
  --entrypoint bash "$IMG" -c '
    set -uo pipefail
    G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
    rc=0
    run() { local name=$1; shift; "$@" > /results/$name.log 2>&1; local s=$?; echo "$name exit=$s"; tail -3 /results/$name.log; [ $s -eq 0 ] || rc=1; }
    export QSA_BF16_K_LOAD=raw
    run contract python3 -B "$G2/test_contract_cpu.py"
    run address python3 -B "$G2/audit_addressing_cpu.py"
    run q-lds python3 -B "$G2/test_q_lds_cpu.py"
    QSA_TRACE_ROWS_PER_BLOCK=16 run trace16 python3 -B "$G2/trace_check_cpu.py"
    QSA_TRACE_ROWS_PER_BLOCK=32 run trace32 python3 -B "$G2/trace_check_cpu.py"
    unset QSA_BF16_K_LOAD
    run k-tdm python3 -B "$G2/test_k_tdm_cpu.py" -v
    run routing python3 -B "$G2/test_bf16_routing_cpu.py" -v
    ISA=/results/new/tdm/isa/qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4
    run forced-tdm-isa python3 -B "$G2/check_tdm_isa_cpu.py" --r16 "${ISA}_r16_c32_flydsl_tdm/21_final_isa.s" --r16-cols 32 --r32 "${ISA}_r32_c64_flydsl_tdm/21_final_isa.s" --json /results/forced-tdm-c32-isa.json
    ISA=/results/new/tdm64/isa/qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4
    run legacy-tdm64-isa python3 -B "$G2/check_tdm_isa_cpu.py" --r16 "${ISA}_r16_c64_flydsl_tdm/21_final_isa.s" --r16-cols 64 --json /results/forced-tdm-c64-isa.json
    run routing-isa python3 -B "$G2/check_bf16_routing_isa_cpu.py" /results/new --base /results/base
    sha256sum /source/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py /source/aiter/ops/flydsl/kernels/mqa_logits/_g2/*.py > /results/source.sha256
    echo "ALL_GATES_RC=$rc"; exit $rc'
