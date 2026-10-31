#!/bin/bash
# Final CPU re-verification of source-matched, previously rebuilt no-device ISA.
set -euo pipefail
SOURCE=$(realpath "${1:?source directory required}")
RESULTS=$(realpath "${2:?compile result directory required}")
expected=$(cut -d ' ' -f1 "$RESULTS/source.sha256" | head -1)
actual=$(sha256sum "$SOURCE/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py" | cut -d ' ' -f1)
test "$actual" = "$expected"
docker run --rm --network none -v "$SOURCE:/source:ro" -v "$RESULTS:/results" -w /source \
  -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
  -e QSA_GOLDEN_DIR=/source/golden_baseline -e QSA_BF16_K_LOAD=raw -e QSA_BF16_R64=0 \
  --entrypoint bash rocm/fw-bringup-mlperf:gfx1250-atom-20261015 -c '
    set -euo pipefail
    test ! -e /dev/kfd; test ! -e /dev/dri
    G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
    python3 -B -c "import importlib.metadata as m; assert m.version(\"flydsl\")==\"0.3.3.dev886\""
    rc=0
    run() { local name=$1; shift; set +e; "$@" > /results/$name.log 2>&1; local s=$?; set -e; echo "$name exit=$s"; tail -5 /results/$name.log; [ $s -eq 0 ] || rc=1; }
    run contract python3 -B "$G2/test_contract_cpu.py"
    run address python3 -B "$G2/audit_addressing_cpu.py"
    run q-lds python3 -B "$G2/test_q_lds_cpu.py"
    QSA_TRACE_ROWS_PER_BLOCK=16 run trace16 python3 -B "$G2/trace_check_cpu.py"
    QSA_TRACE_ROWS_PER_BLOCK=32 run trace32 python3 -B "$G2/trace_check_cpu.py"
    run k-tdm python3 -B "$G2/test_k_tdm_cpu.py" -v
    run routing python3 -B "$G2/test_bf16_routing_cpu.py" -v
    run fp8-k-tdm python3 -B "$G2/test_fp8_k_tdm_cpu.py" -v
    run bm64 python3 -B "$G2/test_bf16_bm64_cpu.py" -v
    ISA=/results/tdm/isa/qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4
    run strict-tdm python3 -B "$G2/check_tdm_isa_cpu.py" --r16 "${ISA}_r16_c32_flydsl_tdm/21_final_isa.s" --r16-cols 32 --r32 "${ISA}_r32_c64_flydsl_tdm/21_final_isa.s" --r64 /results/r64tdm/isa/qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4_r64_c64_flydsl_tdm/21_final_isa.s --json /results/strict-tdm.json
    run strict-legacy python3 -B "$G2/check_tdm_isa_cpu.py" --r16 /results/tdm64/isa/qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4_r16_c64_flydsl_tdm/21_final_isa.s --r16-cols 64 --json /results/strict-legacy.json
    run routing-isa python3 -B "$G2/check_bf16_routing_isa_cpu.py" /results
    run fp8-isa python3 -B "$G2/check_fp8_tdm_isa_cpu.py" /results
    sha256sum "$G2/../qsa_paged_mqa_logits_page4_gfx1250.py" "$G2/"*.py > /results/source.sha256
    echo "ALL_GATES_RC=$rc"; [ $rc -eq 0 ]
    python3 -B "$G2/summarize_merge_cpu.py" /results /results/merge_isa_evidence.json'
