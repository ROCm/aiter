"""Cost of LDS word -> CTA-uniform scalar in Gluon (32 values), variants. One GPU."""
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import vllm.models.minimax_m3.amd.mono.gluon.kernel as K
from vllm.models.minimax_m3.amd.mono.gluon.prims import T1, THREADS, uniform


@gluon.jit
def clk():
    return gl.inline_asm_elementwise("s_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s", [], dtype=gl.int64,
                                     is_pure=False, pack=1)


@gluon.jit
def probe(out_ptr, stamps, V: gl.constexpr, REPS: gl.constexpr):
    bid = gl.program_id(0)
    t = K.ar(THREADS, T1)
    xch = gl.allocate_shared_memory(gl.int32, [64], K.FLAT1)
    R1: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    R64: gl.constexpr = gl.BlockedLayout([64], [64], [8], [0])
    acc = bid * 0
    c0 = clk()
    for r in range(REPS):
        xch.scatter(t + r, t % 64, 0)
        gl.barrier()
        for j in gl.static_range(32):
            if V == 0:
                v = gl.max(xch.slice(j, 1).load(R1), axis=0)
            elif V == 1:
                v = gl.reshape(xch.slice(j, 1).load(R1), [1])
                v = gl.sum(v, axis=0)
            elif V == 2:
                allw = xch.load(R64)
                ii = gl.arange(0, 64, layout=R64)
                v = gl.sum(gl.where(ii == j, allw, 0), axis=0)
            else:
                v = gl.max(xch.slice(j, 1).load(gl.BlockedLayout([1], [1], [1], [0], )) if False else xch.slice(j, 1).load(R1), axis=0)
            acc = acc + uniform(v)
        gl.barrier()
    c1 = clk()
    gl.store(out_ptr + bid + t * 0, acc + t * 0, mask=t == 0)
    gl.store(stamps + bid + t * 0, (c1 - c0) + t * 0, mask=t == 0)


dev = torch.device("cuda")
out = torch.zeros(32, dtype=torch.int32, device=dev)
st = torch.zeros(32, dtype=torch.int64, device=dev)
for V in (0, 1, 2):
    try:
        for _ in range(2):
            probe[(32,)](out, st, V=V, REPS=20, num_warps=8)
        torch.cuda.synchronize()
        print(f"variant {V}: {st.float().median().item() / 100.0 / 20:.2f} us per 32 values")
    except Exception as e:
        print(f"variant {V}: failed {type(e).__name__}: {str(e)[:200]}")
