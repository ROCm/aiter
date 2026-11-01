#!/bin/bash
# Default fast must not change bf16, r16/decode, scalar, or raw ISA.
set -euo pipefail
SOURCE=$(realpath "${1:?source}")
RESULTS=$(realpath -m "${2:?results}")
CHAMPION=$(realpath "${3:?champion results}")
mkdir -p "$RESULTS"
docker run --rm --network none -v "$SOURCE:/source:ro" -v "$RESULTS:/results" -v "$CHAMPION:/champion:ro" -w /source \
 -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
 -e FLYDSL_DUMP_IR=1 -e FLYDSL_DEBUG_DUMP_ASM=1 \
 --entrypoint bash rocm/fw-bringup-mlperf:gfx1250-atom-20261015 -c '
 set -euo pipefail
 test ! -e /dev/kfd; test ! -e /dev/dri
 python3 -B -c "import importlib.metadata as m; assert m.version(\"flydsl\")==\"0.3.3.dev886\""
 for mode in auto raw; do
   export QSA_BF16_K_LOAD=$mode QSA_RESULTS=/results/$mode FLYDSL_DUMP_DIR=/results/$mode/isa FLYDSL_RUNTIME_CACHE_DIR=/results/$mode/cache
   mkdir -p "$QSA_RESULTS"
   python3 -B aiter/ops/flydsl/kernels/mqa_logits/_g2/compile_only.py > "$QSA_RESULTS/compile.log" 2>&1 || { tail -80 "$QSA_RESULTS/compile.log"; exit 1; }
 done
 python3 -B - <<"PY"
from pathlib import Path
import json
checks = []
for mode in ("auto", "raw"):
    root, base = Path("/results") / mode / "isa", Path("/champion") / mode / "isa"
    for source in root.rglob("21_final_isa.s"):
        if "_cvtfast" in str(source):
            continue
        other = base / source.relative_to(root)
        assert source.read_bytes() == other.read_bytes(), str(source)
        checks.append(str(source.relative_to(Path("/results"))))
assert checks
Path("/results/fast-unaffected-identity.json").write_text(json.dumps({"byte_identical": True, "isa_files": checks}, indent=2))
print("FAST_FLAG_UNAFFECTED_ISA_BYTE_IDENTITY", len(checks), "PASS")
PY
'
