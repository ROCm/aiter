#!/bin/bash
# Re-run final CPU tests/strict gates against an existing no-device ISA trace.
set -euo pipefail
ARCHIVE=$(realpath "${1:?source archive required}")
TRACE=$(realpath "${2:?existing trace result directory required}")
STAGE=$(dirname "$ARCHIVE")
SOURCE=$(mktemp -d "$STAGE/source-verify-tdm4.XXXXXX")
RESULTS=$(mktemp -d "$STAGE/verify-tdm4.XXXXXX")
tar -xzf "$ARCHIVE" -C "$SOURCE"
printf 'SOURCE=%s\nTRACE=%s\nRESULTS=%s\n' "$SOURCE" "$TRACE" "$RESULTS"
docker run --rm -v "$SOURCE:/source:ro" -v "$TRACE:/trace" -v "$RESULTS:/results" -w /source \
  -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
  -e QSA_GOLDEN_DIR=/source/golden_baseline -e QSA_BF16_K_LOAD=raw -e QSA_FP8_K_LOAD=raw \
  --entrypoint bash rocm/fw-bringup-mlperf:gfx1250-atom-20261015 -c '
    set -euo pipefail
    test ! -e /dev/kfd; test ! -e /dev/dri
    G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
    expected=$(cut -d " " -f1 /trace/source.sha256)
    actual=$(sha256sum "$G2/../qsa_paged_mqa_logits_page4_gfx1250.py" | cut -d " " -f1)
    test "$actual" = "$expected"
    printf "kernel_sha256=%s\n" "$actual" > /results/source-match.txt
    python3 -B -c "import importlib.metadata as m; assert m.version(\"flydsl\")==\"0.3.3.dev886\"; print(\"stock\",m.version(\"flydsl\"))"
    for test in test_contract_cpu audit_addressing_cpu test_q_lds_cpu test_k_tdm_cpu test_bf16_routing_cpu test_fp8_k_tdm_cpu test_bf16_bm64_cpu; do
      python3 -B "$G2/$test.py" > "/results/$test.log" 2>&1
      tail -5 "/results/$test.log"
    done
    for rows in 16 32; do
      QSA_TRACE_ROWS_PER_BLOCK=$rows python3 -B "$G2/trace_check_cpu.py" > "/results/trace$rows.log" 2>&1
      tail -4 "/results/trace$rows.log"
    done
    python3 -B "$G2/check_bf16_routing_isa_cpu.py" /trace > /results/bf16-isa.log 2>&1
    python3 -B "$G2/check_fp8_tdm_isa_cpu.py" /trace > /results/fp8-isa.log 2>&1
    cp /trace/routing-isa.json /results/bf16-routing-isa.json
    cp /trace/fp8-tdm-isa.json /results/fp8-tdm-isa.json
    python3 -B -c "import json,pathlib; allrec=[json.load(open(pathlib.Path(\"/trace\")/m/\"compile_only.json\"))[\"specializations\"] for m in (\"auto\",\"raw\",\"tdm\")]; assert all(len(r)==7 and all(x[\"compiled\"] for x in r) for r in allrec); print(\"PASS: 21 real-ISA specializations (7 x 3 modes), no device\")"
    sha256sum "$G2/../qsa_paged_mqa_logits_page4_gfx1250.py" "$G2/test_fp8_k_tdm_cpu.py" "$G2/test_bf16_bm64_cpu.py" "$G2/check_tdm_isa_cpu.py" "$G2/check_fp8_tdm_isa_cpu.py" > /results/source.sha256
  '
