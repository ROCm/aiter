#!/bin/bash
# Verify default/fast identity and byte-identical explicit-current rollback ISA.
set -euo pipefail
SOURCE=$(realpath "${1:?source}")
RESULTS=$(realpath "${2:?native results}")
CHAMPION=$(realpath "${3:?champion results}")
docker run --rm --network none -v "$SOURCE:/source:ro" -v "$RESULTS:/results" \
 -v "$CHAMPION:/champion:ro" -w /source \
 --entrypoint bash rocm/fw-bringup-mlperf:gfx1250-atom-20261015 -c '
 set -euo pipefail
 test ! -e /dev/kfd; test ! -e /dev/dri
 P=qsa_paged_mqa_logits_page4_gfx1250_fp8_wmma_D128_H4_P4_r32_c64_flydsl_tdm
 python3 -B aiter/ops/flydsl/kernels/mqa_logits/_g2/check_fp8_cvt_isa_cpu.py \
  --champion /champion/auto/isa/$P/21_final_isa.s \
  --current /results/current/isa/$P/21_final_isa.s \
  --fast /results/fast/isa/${P}_cvtfast/21_final_isa.s \
  --default /results/default/isa/${P}_cvtfast/21_final_isa.s \
  --micro-current /results/current/isa/qsa_fp8cvt_micro_current/21_final_isa.s \
  --micro-fast /results/fast/isa/qsa_fp8cvt_micro_fast/21_final_isa.s \
  --json /results/fp8cvt_isa_evidence.json
'
