"""Launch stage_down alone (loads / mids / epilogue switched off by DBG) on one GPU."""
import os
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import vllm.models.minimax_m3.amd.mono.gluon.kernel as K


@gluon.jit
def p_down(ptab, scratch, peers, errors, ar_out, out, li, S: gl.constexpr, DBG: gl.constexpr):
    bid = gl.program_id(0)
    E = ()
    W = ()
    for i in gl.static_range(4 * S):
        E = E + (bid * 0 + i,)
        W = W + (bid.to(gl.float32) * 0.0 + 0.25,)
    w2 = K.lptr(ptab, li, 14, gl.uint8)
    s2 = K.lptr(ptab, li, 15, gl.uint8)
    ms8 = gl.allocate_shared_memory(gl.uint8, [8, 4, 4, 128], K.FLAT4)
    mc8 = gl.allocate_shared_memory(gl.uint8, [8, 4, 4, 4], K.FLAT4)
    msh = gl.allocate_shared_memory(gl.uint8, [2, 4, 4, 128], K.FLAT4)
    mch = gl.allocate_shared_memory(gl.uint8, [2, 4, 4, 4], K.FLAT4)
    red = gl.allocate_shared_memory(gl.float32, [128, 4], K.FLAT2)
    K.stage_down(bid, li, 57, E, W, w2, s2, ms8, mc8, msh, mch, red, scratch, peers, scratch.to(gl.int64, bitcast=True),
                 li.to(gl.int64), 0, ar_out, li + 1, errors, bid, 1.0, S, 4096, DBG)
    t = K.tids()
    gl.store(out + bid * 512 + t, t)


S = int(os.environ.get("S", "1"))
DBG = int(os.environ.get("DBG", str(512 | 1024 | 2048)))
dev = "cuda"
w2 = torch.zeros(129 * 6144 * 384, dtype=torch.uint8, device=dev)
s2 = torch.full((129 * 6144 * 24,), 127, dtype=torch.uint8, device=dev)
ptab = torch.zeros(20, dtype=torch.int64, device=dev)
ptab[14] = w2.data_ptr()
ptab[15] = s2.data_ptr()
scratch = torch.zeros(4 << 20, dtype=torch.uint8, device=dev)
peers = torch.tensor([scratch.data_ptr()] * 4, dtype=torch.int64, device=dev)
errors = torch.zeros(256, dtype=torch.int32, device=dev)
ar_out = torch.zeros(16, 6144, dtype=torch.bfloat16, device=dev)
out = torch.zeros(256 * 512, dtype=torch.int32, device=dev)
for it in range(int(os.environ.get("ITERS", "3"))):
    cc = p_down[(192,)](ptab, scratch, peers, errors, ar_out, out, 0, S=S, DBG=DBG, num_warps=8,
                        enable_fp_fusion=False)
    torch.cuda.synchronize()
print("ok", S, DBG, "out check", bool((out.view(256, 512)[:192] == torch.arange(512, device=dev)).all()))
