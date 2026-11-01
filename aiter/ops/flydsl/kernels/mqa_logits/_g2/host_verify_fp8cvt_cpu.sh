#!/bin/bash
# All champion CPU/ISA gates plus native-conversion proof, no GPU devices.
set -euo pipefail
SOURCE=$(realpath "${1:?source}")
RESULTS=$(realpath -m "${2:?results}")
GOLDEN=$(realpath "${3:?read-only golden}")
CHAMPION=$(realpath "${4:?champion ISA results}")
mkdir -p "$RESULTS"
docker run --rm --network none -v "$SOURCE:/source:ro" -v "$RESULTS:/results" \
 -v "$GOLDEN:/golden:ro" -v "$CHAMPION:/champion:ro" -w /source \
 -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
 -e QSA_GOLDEN_DIR=/golden -e FLYDSL_DUMP_IR=1 -e FLYDSL_DEBUG_DUMP_ASM=1 \
 -e QSA_SKIP_COMPILE=${QSA_SKIP_COMPILE:-0} \
 --entrypoint bash rocm/fw-bringup-mlperf:gfx1250-atom-20261015 -c '
 set -euo pipefail
 test ! -e /dev/kfd; test ! -e /dev/dri
 G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
 python3 -B -c "import importlib.metadata as m; assert m.version(\"flydsl\")==\"0.3.3.dev886\""
 if [ "$QSA_SKIP_COMPILE" = 1 ]; then
   expected=$(head -1 /results/source.sha256 | cut -d " " -f1)
   actual=$(sha256sum /source/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py | cut -d " " -f1)
   test "$expected" = "$actual"
   echo "REUSE source-hash-matched no-device ISA"
 else
 for mode in auto off raw tdm p2 p4 p8; do
   unset QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS QSA_BF16_R64 QSA_BF16_PERSIST QSA_BF16_K_LOAD QSA_G2_ACCESS QSA_FP8_CVT
   export QSA_FP8_CVT=current  # Explicit current must retain all legacy ISA.
    case "$mode" in
     auto) :;;
     off) export QSA_BF16_PERSIST=0;;
     raw|tdm) export QSA_BF16_K_LOAD=$mode QSA_BF16_PERSIST=0;;
     p2|p4|p8) export QSA_BF16_PERSIST=1 QSA_PERSIST_COLS=${mode#p};;
   esac
   export QSA_RESULTS=/results/$mode FLYDSL_DUMP_DIR=/results/$mode/isa FLYDSL_RUNTIME_CACHE_DIR=/results/$mode/cache
   mkdir -p "$QSA_RESULTS"
   python3 -B "$G2/compile_only.py" > "$QSA_RESULTS/compile.log" 2>&1 || { tail -80 "$QSA_RESULTS/compile.log"; exit 1; }
   echo "COMPILE mode=$mode PASS"
 done
 fi
 python3 -B - <<"PY"
from pathlib import Path
import json
r, c = Path("/results"), Path("/champion")
checks = []
for mode in ("auto", "off", "raw", "tdm", "p2", "p4", "p8"):
    paths = list((c / mode / "isa").rglob("21_final_isa.s"))
    assert paths
    for before in paths:
        after = r / mode / "isa" / before.relative_to(c / mode / "isa")
        assert before.read_bytes() == after.read_bytes(), str(after)
        checks.append(str(after.relative_to(r)))
(r / "switch-off-identity.json").write_text(json.dumps({"byte_identical": True, "isa_files": checks}, indent=2))
print("ALL_SWITCH_OFF_ISA_BYTE_IDENTITY", len(checks), "PASS")
PY
 unset QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS FLYDSL_DUMP_IR FLYDSL_DEBUG_DUMP_ASM QSA_FP8_CVT
 export QSA_FP8_CVT=current QSA_BF16_K_LOAD=raw QSA_BF16_R64=0 QSA_BF16_PERSIST=0
 rc=0
 run() { local name=$1; shift; set +e; "$@" > /results/$name.log 2>&1; local s=$?; set -e; echo "$name exit=$s"; tail -4 /results/$name.log; [ $s -eq 0 ] || rc=1; }
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
 run fp8-cvt python3 -B "$G2/test_fp8_cvt_cpu.py" -v
 ISA=/results/tdm/isa/qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4
 run strict-tdm python3 -B "$G2/check_tdm_isa_cpu.py" --r16 "${ISA}_r16_c32_flydsl_tdm/21_final_isa.s" --r16-cols 32 --r32 "${ISA}_r32_c64_flydsl_tdm/21_final_isa.s" --json /results/strict-tdm.json
 run routing-isa python3 -B "$G2/check_bf16_routing_isa_cpu.py" /results
 run fp8-isa python3 -B "$G2/check_fp8_tdm_isa_cpu.py" /results --cvt current
 run persist-isa python3 -B "$G2/summarize_persist_cpu.py" /results "$G2/merge_isa_evidence.json" /results/persist_isa_evidence.json
 sha256sum /source/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py /source/aiter/ops/flydsl/kernels/mqa_logits/_g2/*.py > /results/source.sha256
 echo "ALL_GATES_RC=$rc"; exit $rc'
