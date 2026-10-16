import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

L = gl.constexpr(gl.BlockedLayout([4], [64], [8], [0]))

@gluon.jit
def opq():
    return gl.multiple_of(gl.inline_asm_elementwise("s_mov_b32 $0, 0", "=s", [], dtype=gl.int32, is_pure=False,
                                                    pack=1), 65536)

@gluon.jit
def k(x, n):
    for i in range(n):
        idx = gl.arange(0, 2048, layout=L) + opq()
        v = gl.load(x + idx)
        gl.store(x + idx + 2048 * (i + 1), v + 1)

x = torch.zeros(1 << 16, dtype=torch.int32, device="cuda")
cc = k.warmup(x, 3, grid=(1,), num_warps=8)
cc._init_handles()
a = cc.asm["amdgcn"]
print("ok dwordx4 loads", a.count("global_load_dwordx4") + a.count("buffer_load_dwordx4"), "s_mov 0 in loop", a.count("s_mov_b32 s"), cc.n_regs)
