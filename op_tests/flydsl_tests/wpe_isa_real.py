"""Generate the REAL final ISA with the waves_per_eu backend opt actually
applied (the FLYDSL_DUMP_IR path uses opts="" so it can't show the hint).

We re-run gpu-module-to-binary{format=isa opts="--amdgpu-waves-per-eu=N"} on the
post-reconcile MLIR (stage 18) that the dump already wrote, for N in {0,4}, and
compare .vgpr_count. If the opt is honored, the VGPR cap (hence count) differs.
"""
import sys
from pathlib import Path
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler.jit_function import _extract_isa_text, _create_mlir_context

asm_path = Path(sys.argv[1])  # 18_reconcile_unrealized_casts.mlir
wpe = int(sys.argv[2])
opt = f"--amdgpu-waves-per-eu={wpe}" if wpe else ""

asm = asm_path.read_text()
ctx = _create_mlir_context()
with ctx:
    ctx.allow_unregistered_dialects = True
    mod = ir.Module.parse(asm, context=ctx)
    pm = PassManager.parse(
        f'builtin.module(gpu-module-to-binary{{format=isa opts="{opt}" section= toolkit=}})',
        context=ctx,
    )
    pm.run(mod.operation)
    raw = mod.operation.get_asm(enable_debug_info=False)
    isa = _extract_isa_text(raw)
    out = asm_path.parent / f"real_isa_wpe{wpe}.s"
    out.write_text(isa)
    for line in isa.splitlines():
        if any(k in line for k in (".vgpr_count", ".vgpr_spill_count", ".sgpr_count", "amdhsa_next_free_vgpr")):
            print(line.strip())
    print(f"-> {out}")
