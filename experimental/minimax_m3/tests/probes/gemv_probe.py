import os, sys, torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4
from vllm.models.minimax_m3.amd.mono.gluon.kernel import (W8_L, A_L, B_L, MFMA, FLAT3, FLAT2, w8_tile, w8_rows,
    unit_scales, bd_reduce, PR_L)

@gluon.jit
def p_load(w, out, MODE: gl.constexpr):
    m = w8_rows()
    a = w8_tile(w, (m // 16) * 2048 + (m % 16) * 16)
    if MODE == 0:
        gl.store(out + gl.arange(0, 128, layout=gl.SliceLayout(1, A_L))[:, None] * 128
                 + gl.arange(0, 128, layout=gl.SliceLayout(0, A_L))[None, :], a)
    else:
        xs = gl.allocate_shared_memory(gl.uint8, [2, 8, 16, 128], FLAT3)
        z = gl.zeros([8, 16, 128], gl.uint8, gl.BlockedLayout([1, 1, 16], [1, 8, 8], [8, 1, 1], [2, 1, 0]))
        xs.index(0).store(z)
        xs.index(1).store(z)
        gl.barrier()
        bv = xs.index(1).reshape([128, 128]).permute((1, 0)).load(B_L)
        acc = gl.zeros([128, 128], gl.float32, MFMA)
        acc = cdna4.mfma_scaled(a.to(gl.float8e4nv, bitcast=True), unit_scales([128, 4], A_L), "e4m3",
                                bv.to(gl.float8e4nv, bitcast=True), unit_scales([128, 4], B_L), "e4m3", acc)
        if MODE == 2:
            red = gl.allocate_shared_memory(gl.float32, [128, 128], FLAT2)
            los, his = bd_reduce(acc, 1, 8, red)
            j = gl.arange(0, 8, layout=gl.SliceLayout(1, PR_L))[:, None]
            n = gl.arange(0, 16, layout=gl.SliceLayout(0, PR_L))[None, :]
            gl.store(out.to(gl.pointer_type(gl.float32), bitcast=True) + j * 16 + n, los[0] + his[0])
        else:
            gl.store(out.to(gl.pointer_type(gl.float32), bitcast=True)
                     + gl.arange(0, 128, layout=gl.SliceLayout(1, MFMA))[:, None] * 128
                     + gl.arange(0, 128, layout=gl.SliceLayout(0, MFMA))[None, :], acc)

w = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
o = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
for mode in [int(x) for x in sys.argv[1:]]:
    try:
        cc = p_load.warmup(w, o, MODE=mode, grid=(1,), num_warps=8)
        print("mode", mode, "ok", cc.asm["amdgcn"].count("buffer_load_dwordx4"), flush=True)
    except Exception as e:
        print("mode", mode, "FAIL", str(e)[:300], flush=True)
