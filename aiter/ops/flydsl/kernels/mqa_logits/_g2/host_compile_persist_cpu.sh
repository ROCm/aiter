#!/bin/bash
# Round PERSIST: compile-only CPU container (no GPU devices, no dispatch).
# Modes: truly unset auto (default P8), off (QSA_BF16_PERSIST=0),
# raw/tdm (persistent forced OFF), and explicit p2/p4/p8.
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
    python3 -B -c "import importlib.metadata as m; assert m.version(\"flydsl\")==\"0.3.3.dev886\"; print(\"flydsl ok\")"
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
      start=$(date +%s)
      python3 -B "$G2/compile_only.py" > "$QSA_RESULTS/compile.log" 2>&1 || { tail -80 "$QSA_RESULTS/compile.log"; exit 1; }
      grep "^\[COMPILED\]\|^\[FAILED\]" "$QSA_RESULTS/compile.log"
      echo "mode=$mode seconds=$(( $(date +%s) - start ))"
    done
    sha256sum /source/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py /source/aiter/ops/flydsl/kernels/mqa_logits/_g2/*.py > /results/source.sha256
    echo COMPILE_RC=0'
