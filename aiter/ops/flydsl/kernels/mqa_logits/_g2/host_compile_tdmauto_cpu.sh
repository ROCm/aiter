#!/bin/bash
# CPU-only stock-FlyDSL compile and routing gate. No GPU devices are exposed.
# Usage: bash host_compile_tdmauto_cpu.sh /absolute/staging/source.tar.gz
set -euo pipefail
ARCHIVE=$(realpath "${1:?source archive required}")
STAGE=$(dirname "$ARCHIVE")
IMG=rocm/fw-bringup-mlperf:gfx1250-atom-20261015
SOURCE=$(mktemp -d "$STAGE/source-tdmauto.XXXXXX")
RESULTS=$(mktemp -d "$STAGE/results-tdmauto.XXXXXX")
tar -xzf "$ARCHIVE" -C "$SOURCE"
printf 'SOURCE=%s\nRESULTS=%s\n' "$SOURCE" "$RESULTS"
docker run --rm -v "$SOURCE:/source:ro" -v "$RESULTS:/results" -w /source \
  -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
  -e QSA_GOLDEN_DIR=/source/golden_baseline \
  -e FLYDSL_DUMP_IR=1 -e FLYDSL_DEBUG_DUMP_ASM=1 \
  --entrypoint bash "$IMG" -c '
    set -euo pipefail
    G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
    python3 -B -c "import importlib.metadata as m; assert m.version(\"flydsl\")==\"0.3.3.dev886\"; print(\"FlyDSL\", m.version(\"flydsl\"))"
    for mode in auto raw tdm; do
      export QSA_BF16_K_LOAD=$mode QSA_RESULTS=/results/$mode
      export FLYDSL_DUMP_DIR=/results/$mode/isa FLYDSL_RUNTIME_CACHE_DIR=/results/$mode/cache
      mkdir -p "$QSA_RESULTS"
      python3 -B "$G2/compile_only.py" > "$QSA_RESULTS/compile.log" 2>&1
      tail -45 "$QSA_RESULTS/compile.log"
    done
    export QSA_BF16_K_LOAD=raw
    python3 -B "$G2/test_contract_cpu.py" > /results/contract.log 2>&1
    python3 -B "$G2/audit_addressing_cpu.py" > /results/address.log 2>&1
    python3 -B "$G2/test_q_lds_cpu.py" > /results/q-lds.log 2>&1
    QSA_TRACE_ROWS_PER_BLOCK=16 python3 -B "$G2/trace_check_cpu.py" > /results/trace16.log 2>&1
    QSA_TRACE_ROWS_PER_BLOCK=32 python3 -B "$G2/trace_check_cpu.py" > /results/trace32.log 2>&1
    python3 -B "$G2/test_k_tdm_cpu.py" -v > /results/k-tdm.log 2>&1
    python3 -B "$G2/test_bf16_routing_cpu.py" -v > /results/routing.log 2>&1
    ISA=/results/tdm/isa/qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4
    python3 -B "$G2/check_tdm_isa_cpu.py" --r16 "${ISA}_r16_c64_flydsl_tdm/21_final_isa.s" --r32 "${ISA}_r32_c64_flydsl_tdm/21_final_isa.s" --json /results/forced-tdm-isa.json
    ISA=/results/auto/isa/qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4
    python3 -B "$G2/check_tdm_isa_cpu.py" --r32 "${ISA}_r32_c64_flydsl_tdm/21_final_isa.s" --json /results/auto-r32-isa.json
    python3 -B "$G2/check_bf16_routing_isa_cpu.py" /results > /results/routing-isa.log 2>&1
    for log in contract address q-lds trace16 trace32 k-tdm routing; do tail -5 "/results/$log.log"; done
    sha256sum /source/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py /source/aiter/ops/flydsl/kernels/mqa_logits/_g2/test_bf16_routing_cpu.py > /results/source.sha256
  '
