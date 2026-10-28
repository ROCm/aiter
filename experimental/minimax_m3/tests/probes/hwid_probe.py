"""Which XCC / SE / CU each CTA of a 256-CTA launch lands on (s_getreg HW_ID / XCC_ID)."""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

L = gl.constexpr(gl.BlockedLayout([1], [64], [8], [0]))


@gluon.jit
def k(out):
    bid = gl.program_id(0)
    t = gl.arange(0, 512, layout=L)
    hw = gl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(HW_REG_HW_ID)", "=s,v", [t], dtype=gl.int32,
                                   is_pure=False, pack=1)
    xc = gl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s,v", [t], dtype=gl.int32,
                                   is_pure=False, pack=1)
    gl.store(out + bid * 2 + t, gl.where(t == 0, hw, xc), mask=t < 2)


out = torch.zeros(256 * 2, dtype=torch.int32, device="cuda")
k[(256,)](out, num_warps=8)
torch.cuda.synchronize()
o = out.view(256, 2).cpu()
print(o[:12].tolist())
print([hex(v) for v in o[:12, 0].tolist()], [hex(v) for v in o[:12, 1].tolist()])
asm = k.warmup(out, num_warps=8, grid=(256,)).asm["amdgcn"]
print([l for l in asm.splitlines() if "getreg" in l])
