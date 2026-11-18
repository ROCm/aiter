"""What does the hardware fp4 pack (cvt_scalef32_pk_fp4_f32) do with a value / scale? Against round-to-nearest-even."""
import torch, flydsl.compiler as flyc, flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import Int64, T
from flydsl._mlir.dialects import llvm
import hxdev as hx
I32, F32, i64 = fx.Int32, fx.Float32, fx.Int64
@flyc.kernel(known_block_size=[256, 1, 1])
def body(src: Int64, dst: Int64, sc: Int64):
    t = hx.opaque_v(fx.thread_idx.x); b = hx.opaque_s(fx.block_idx.x)
    i = b * 256 + t
    a = hx.gload(src + i64(i * 8), F32); bb = hx.gload(src + i64(i * 8 + 4), F32); s = hx.gload(sc + i64(i * 4), F32)
    p = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, hx._v(I32(0)), hx._v(a), hx._v(bb), hx._v(s), 0)
    hx.gstore(dst + i64(i * 4), I32(p))
@flyc.jit
def launch(src: Int64, dst: Int64, sc: Int64, stream: fx.Stream = fx.Stream(None)):
    body(src, dst, sc).launch(grid=(64,), block=(256,), stream=stream)
dev = "cuda"; N = 64 * 256
v = torch.cat([torch.linspace(-7, 7, N), torch.randn(N) * 2.5]).to(dev).view(N, 2).contiguous()      # pairs (a, b)
v = torch.stack([torch.linspace(-7, 7, N), torch.linspace(0, 6.6, N)], 1).to(dev).contiguous()
scb = torch.full((N,), 127, dtype=torch.int32, device=dev)                                            # scale as the kernels pass it: (e8m0 << 23) as f32 = 2^(e8m0 - 127)
scale = (scb << 23).view(torch.float32).contiguous()
out = torch.zeros(N, dtype=torch.int32, device=dev)
launch(v.data_ptr(), out.data_ptr(), scale.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
TAB = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
got = torch.stack([TAB[(out & 15).long()], TAB[((out >> 4) & 15).long()]], 1)
def rne(y):
    ay = y.abs().clamp(max=6.0); inv = torch.where(ay < 2, torch.full_like(ay, 2.0), torch.where(ay < 4, torch.ones_like(ay), torch.full_like(ay, 0.5)))
    return (torch.round(ay * inv) / inv).clamp(max=6.0) * y.sign()
exp = rne(v)
bad = (got.abs() != exp.abs()) | ((got != 0) & (got.sign() != exp.sign()))
print("scale 1.0: %d values, mismatches vs round-to-nearest-even %d" % (2 * N, int(bad.sum())))
if bad.any():
    idx = bad.nonzero()[:14]
    for r, c in idx.tolist(): print("   v %.5f -> hardware %.2f, RNE %.2f" % (float(v[r, c]), float(got[r, c]), float(exp[r, c])))
# exact ties
ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 6.5, 100.0, -0.25, -0.75, -2.5, -5.0], device=dev)
v2 = torch.zeros(N, 2, device=dev); v2[:len(ties), 0] = ties; out.zero_()
launch(v2.data_ptr(), out.data_ptr(), scale.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
print("ties:", [(float(a), float(TAB[int(o) & 15])) for a, o in zip(ties, out[:len(ties)])])
# other scales: is the scale operand the divisor 2^(e - 127)?
for e in (125, 130):
    sc2 = (torch.full((N,), e, dtype=torch.int32, device=dev) << 23).view(torch.float32).contiguous(); out.zero_()
    launch(v.data_ptr(), out.data_ptr(), sc2.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
    got = torch.stack([TAB[(out & 15).long()], TAB[((out >> 4) & 15).long()]], 1)
    for name, y in (("v / 2^(e-127)", v / 2.0 ** (e - 127)), ("v * 2^(e-127)", v * 2.0 ** (e - 127))):
        print("e8m0 %d: mismatches vs RNE(%s): %d" % (e, name, int((got != rne(y)).sum() - ((got == 0) & (rne(y) == 0)).sum() * 0)))
