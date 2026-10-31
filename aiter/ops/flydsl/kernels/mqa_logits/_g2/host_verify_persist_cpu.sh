#!/bin/bash
# Round PERSIST: compile-only ISA (no devices) + every CPU gate + ISA summary.
# usage: host_verify_persist_cpu.sh SOURCE_ARCHIVE  (archive holds the repo
# subtree aiter/ops/flydsl/kernels/mqa_logits and golden_baseline/)
set -euo pipefail
ARCHIVE=$(realpath "${1:?source archive required}")
STAGE=$(dirname "$ARCHIVE")
SOURCE=$(mktemp -d "$STAGE/source-persist.XXXXXX")
RESULTS=$(mktemp -d "$STAGE/results-persist.XXXXXX")
tar -xzf "$ARCHIVE" -C "$SOURCE"
printf 'SOURCE=%s\nRESULTS=%s\n' "$SOURCE" "$RESULTS"
docker run --rm --network none -v "$SOURCE:/source:ro" -v "$RESULTS:/results" -w /source \
  -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
  -e QSA_GOLDEN_DIR=/source/golden_baseline -e FLYDSL_DUMP_IR=1 -e FLYDSL_DEBUG_DUMP_ASM=1 \
  --entrypoint bash rocm/fw-bringup-mlperf:gfx1250-atom-20261015 -c '
    set -euo pipefail
    test ! -e /dev/kfd; test ! -e /dev/dri
    G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
    python3 -B -c "import importlib.metadata as m; assert m.version(\"flydsl\")==\"0.3.3.dev886\"; print(\"flydsl 0.3.3.dev886\")"
    for mode in auto off raw tdm p2 p4 p8; do
      unset QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS QSA_BF16_R64 QSA_BF16_PERSIST QSA_BF16_K_LOAD QSA_G2_ACCESS
      case "$mode" in
        auto) :;; # Truly unset production defaults: auto + persistent P8.
        off) export QSA_BF16_PERSIST=0;;
        raw|tdm) export QSA_BF16_K_LOAD=$mode QSA_BF16_PERSIST=0;;
        p2|p4|p8) export QSA_BF16_PERSIST=1 QSA_PERSIST_COLS=${mode#p};;
      esac
      export QSA_RESULTS=/results/$mode FLYDSL_DUMP_DIR=/results/$mode/isa FLYDSL_RUNTIME_CACHE_DIR=/results/$mode/cache
      mkdir -p "$QSA_RESULTS"
      python3 -B "$G2/compile_only.py" > "$QSA_RESULTS/compile.log" 2>&1 || { tail -80 "$QSA_RESULTS/compile.log"; exit 1; }
      grep "^\[COMPILED\]\|^\[FAILED\]" "$QSA_RESULTS/compile.log"
    done
    unset QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS FLYDSL_DUMP_IR FLYDSL_DEBUG_DUMP_ASM
    export QSA_BF16_K_LOAD=raw QSA_BF16_R64=0 QSA_BF16_PERSIST=0
    rc=0
    run() { local name=$1; shift; set +e; "$@" > /results/$name.log 2>&1; local s=$?; set -e; echo "$name exit=$s"; tail -3 /results/$name.log; [ $s -eq 0 ] || rc=1; }
    run contract python3 -B "$G2/test_contract_cpu.py"
    run address python3 -B "$G2/audit_addressing_cpu.py"
    run q-lds python3 -B "$G2/test_q_lds_cpu.py"
    QSA_TRACE_ROWS_PER_BLOCK=16 run trace16 python3 -B "$G2/trace_check_cpu.py"
    QSA_TRACE_ROWS_PER_BLOCK=32 run trace32 python3 -B "$G2/trace_check_cpu.py"
    run k-tdm python3 -B "$G2/test_k_tdm_cpu.py" -v
    run routing python3 -B "$G2/test_bf16_routing_cpu.py" -v
    run fp8-k-tdm python3 -B "$G2/test_fp8_k_tdm_cpu.py" -v
    run bm64 python3 -B "$G2/test_bf16_bm64_cpu.py" -v
    run persist python3 -B "$G2/test_persist_cpu.py" -v
    ISA=/results/tdm/isa/qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4
    run strict-tdm python3 -B "$G2/check_tdm_isa_cpu.py" --r16 "${ISA}_r16_c32_flydsl_tdm/21_final_isa.s" --r16-cols 32 --r32 "${ISA}_r32_c64_flydsl_tdm/21_final_isa.s" --json /results/strict-tdm.json
    run routing-isa python3 -B "$G2/check_bf16_routing_isa_cpu.py" /results
    run fp8-isa python3 -B "$G2/check_fp8_tdm_isa_cpu.py" /results
    run persist-isa python3 -B "$G2/summarize_persist_cpu.py" /results "$G2/merge_isa_evidence.json" /results/persist_isa_evidence.json
    sha256sum /source/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py /source/aiter/ops/flydsl/kernels/mqa_logits/_g2/*.py > /results/source.sha256
    echo "ALL_GATES_RC=$rc"; exit $rc'
