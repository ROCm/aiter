"""Check the gfx1250 scaled WMMA with A = MXFP4 (aiter per_1x32_f4_quant), B = MXFP8: operand layout by experiment."""
import sys, torch, flydsl.compiler as flyc, flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import Int64
from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from aiter.ops.quant import per_1x32_f4_quant
F32, I32 = fx.Float32, fx.Int32

def wmma_fp4(a, sa, b, sb, c):
    r = rocdl.wmma_scale_f32_16x16x128_f8f6f4(hx.vty(8, hx.T.f32), hx._v(a), hx._v(b), hx._v(c), hx._v(I32(sa)), hx._v(I32(sb)), fmtA=4, fmtB=0)
    return fx.Vector(r, (8,), F32)

@flyc.kernel(known_block_size=[32, 1, 1])
def body(a: Int64, sa: Int64, b: Int64, sb: Int64, o: Int64):
    l = fx.thread_idx.x
    a16 = hx.gload(a + fx.Int64(l * 64), I32, 16)
    bv = hx.gload(b + fx.Int64(l * 64), I32, 16)
    s1 = hx.gload(sa + fx.Int64((l % 16) * 4), I32)
    s2 = hx.gload(sb + fx.Int64((l % 16) * 4), I32)
    r = wmma_fp4(a16, s1, bv, s2, fx.Vector.filled(8, 0.0, F32))
    hx.gstore(o + fx.Int64(l * 32), r)

@flyc.jit
def launch(a: Int64, sa: Int64, b: Int64, sb: Int64, o: Int64, stream: fx.Stream = fx.Stream(None)):
    body(a, sa, b, sb, o).launch(grid=(1,), block=(32,), stream=stream)

dev = "cuda"
s1 = torch.full((16, 4), 127, dtype=torch.uint8, device=dev); xs = torch.full((16, 4), 127, dtype=torch.uint8, device=dev)
def run(A32, xk):                                  # xk [128] values, same for every token
    xq = xk.view(1, 128).expand(16, 128).contiguous().to(torch.float8_e4m3fn)
    B = xq.view(torch.uint8).view(16, 4, 2, 16).permute(2, 0, 1, 3).contiguous()
    o = torch.zeros(32, 8, device=dev)
    launch(A32.data_ptr(), s1.data_ptr(), B.data_ptr(), xs.data_ptr(), o.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
    return o.view(2, 16, 8)[:, 0, :].reshape(16)   # token 0: [rowhalf*8+e]
A32 = torch.zeros(32, 64, dtype=torch.uint8, device=dev); A32[:, :32] = 0x22       # every weight 1.0
base = run(A32, torch.ones(128, device=dev))
ks = torch.arange(128, device=dev)
for i in range(4):
    s1.fill_(127); s1[:, i] = 128
    hit = [blk for blk in range(4) if float(run(A32, ((ks // 32) == blk).float())[0]) > 33]
    print("scale byte %d doubles K blocks %s" % (i, hit))
