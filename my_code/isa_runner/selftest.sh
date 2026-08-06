#!/usr/bin/env bash
# End-to-end check of the ISA runner. Run inside the gfx1250 container:
#   cd <repo>/my_code/isa_runner && bash selftest.sh
set -u
cd "$(dirname "$0")" || exit 1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

GEMM1=../isa_cmp/w1/gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1/21_final_isa.s
rc=0

echo "===== 0. adapter: syntax + CPU-only helper tests ====="
python -m py_compile isa_runner.py tdm_adapter.py test_tdm_adapter.py || rc=1
python -m unittest -v test_tdm_adapter.py || rc=1

echo "===== 1. smoke: assemble + disassemble + order ====="
python isa_runner.py smoke_gfx1250.s --json --disasm smoke.dis || rc=1

echo "===== 2+3. smoke: load, launch, sentinel check, HIP-event bench ====="
timeout 300 python isa_runner.py smoke_gfx1250.s --smoke \
  --iters "${ITERS:-200}" --json --out smoke_bench.json || rc=1

echo "===== 4. real gemm1: reassemble + order + load + symbol ====="
timeout 360 python isa_runner.py "$GEMM1" --json || rc=1
timeout 360 python - "$GEMM1" <<'PY' || rc=1
import json, sys
from isa_runner import IsaModule, build, TDM_GEMM1_BLOCK, TDM_GEMM1_LDS_BYTES
res = build(sys.argv[1])
name = res.kernels[0]
with IsaModule(res.code_object) as mod:
    fn = mod.function(name)
    print(json.dumps({
        "kernel": name,
        "module_loaded": True,
        "function_handle_nonnull": bool(fn.value),
        "recorded_launch_profile": {"block": list(TDM_GEMM1_BLOCK),
                                    "dynamic_lds_bytes": TDM_GEMM1_LDS_BYTES},
        "note": "module-only check; tdm_adapter.py handles production capture",
    }, indent=2))
PY

echo
[ "$rc" -eq 0 ] && echo "SELFTEST: PASS" || echo "SELFTEST: FAIL"
exit "$rc"
