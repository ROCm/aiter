"""Down-stage B operand by LDS gather: 128-B lines (row * 6 + chunk) of a compact MXFP8 mid
store, int64 words gathered per (wave, column) -> bytes -> the batched MFMA's B (B3) without
a layout conversion; e2m1 A x e4m3 B with scales, vs torch. Also the up/gate B3 from an LDS
tile [6 rounds, 8 waves, 16 columns, 128 B] (round i, wave w = chunk 6 w + i)."""
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4

MF3 = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[8, 1, 1]))
A3 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MF3, k_width=16))
B3 = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MF3, k_width=16))
SL3 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[], lane_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]],
                                              warp_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0]], block_bases=[], shape=[8, 16, 4]))
# [128 = (wave, column), 16 int64 of a 128-B line]: lane = 16 (k chunk % 4) + column, regs: the
# chunk's 2 words, + chunk 4
G64 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 8]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 2], [0, 4]],
                                              warp_bases=[[16, 0], [32, 0], [64, 0]], block_bases=[], shape=[128, 16]))
# [128 = (wave, column), 4 codes]: SL3 flattened
G8 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]],
                                             warp_bases=[[16, 0], [32, 0], [64, 0]], block_bases=[], shape=[128, 4]))
FL2 = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0]))
FL4 = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[3, 2, 1, 0]))
FL3 = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[2, 1, 0]))


@gluon.jit
def bytes8(v):
    """int64 [..] -> uint8 [.., 8] (byte j at index j)."""
    b = ()
    for j in gl.static_range(8):
        b = b + (((v >> (8 * j)) & 255).to(gl.uint8),)
    x = gl.join(gl.join(b[0], b[4]), gl.join(b[2], b[6]))
    y = gl.join(gl.join(b[1], b[5]), gl.join(b[3], b[7]))
    return gl.reshape(gl.join(x, y), v.shape + [8])


@gluon.jit
def k_down(a_ptr, sa_ptr, lines_ptr, lsc_ptr, rows_ptr, c_ptr, NL: gl.constexpr):
    # the line store -> LDS
    LL: gl.constexpr = gl.BlockedLayout([1, 2], [32, 2], [8, 1], [1, 0])
    li = gl.arange(0, NL, layout=gl.SliceLayout(1, LL))[:, None]
    lj = gl.arange(0, 16, layout=gl.SliceLayout(0, LL))[None, :]
    st = gl.allocate_shared_memory(gl.int64, [NL, 16], FL2, gl.load(lines_ptr + li * 16 + lj))
    SC: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [8, 1], [1, 0])
    si = gl.arange(0, NL, layout=gl.SliceLayout(1, SC))[:, None]
    sj = gl.arange(0, 4, layout=gl.SliceLayout(0, SC))[None, :]
    ss = gl.allocate_shared_memory(gl.uint8, [NL, 4], FL2, gl.load(lsc_ptr + si * 4 + sj))
    gl.barrier()
    # (wave w, column n) reads line rows[n] * 6 + chunk w % 6
    wn = gl.arange(0, 128, layout=gl.SliceLayout(1, G64))[:, None]
    k8 = gl.arange(0, 16, layout=gl.SliceLayout(0, G64))[None, :]
    line = gl.load(rows_ptr + (wn % 16)) * 6 + (wn // 16) % 6
    v = st.gather(line + k8 * 0, 0)
    bv = gl.reshape(gl.permute(gl.reshape(bytes8(v), [8, 16, 128]), (0, 2, 1)), [8, 128, 16])
    b = gl.convert_layout(bv, B3, assert_trivial=True)
    wn8 = gl.arange(0, 128, layout=gl.SliceLayout(1, G8))[:, None]
    g8 = gl.arange(0, 4, layout=gl.SliceLayout(0, G8))[None, :]
    line8 = gl.load(rows_ptr + (wn8 % 16)) * 6 + (wn8 // 16) % 6
    sb = gl.convert_layout(gl.reshape(ss.gather(line8 + g8 * 0, 0), [8, 16, 4]), SL3, assert_trivial=True)
    w = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, A3)))[:, None, None]
    r = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, A3)))[None, :, None]
    kk = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, A3)))[None, None, :]
    a = gl.load(a_ptr + w * 1024 + r * 64 + kk)
    ws = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, SL3)))[:, None, None]
    rs = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, SL3)))[None, :, None]
    gs = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, SL3)))[None, None, :]
    sa = gl.load(sa_ptr + ws * 64 + rs * 4 + gs)
    acc = cdna4.mfma_scaled(a, sa, "e2m1", b.to(gl.float8e4nv, bitcast=True), sb, "e4m3",
                            gl.zeros([8, 16, 16], gl.float32, MF3))
    OL: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0])
    accb = gl.convert_layout(acc, OL)
    w3 = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, OL)))[:, None, None]
    r3 = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, OL)))[None, :, None]
    n3 = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, OL)))[None, None, :]
    gl.store(c_ptr + w3 * 256 + r3 * 16 + n3, accb)


@gluon.jit
def k_ug(a_ptr, sa_ptr, x_ptr, xs_ptr, c_ptr, I: gl.constexpr):
    # xn [16 columns, 48 chunks, 128 B] -> LDS [6 rounds, 8 waves, 16, 128] (chunk 6 w + i at [i, w])
    XL: gl.constexpr = gl.BlockedLayout([1, 1, 16], [1, 8, 8], [8, 1, 1], [2, 1, 0])
    w4 = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL)))[:, None, None]
    n4 = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, XL)))[None, :, None]
    b4 = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(1, XL)))[None, None, :]
    xa = gl.allocate_shared_memory(gl.uint8, [4, 8, 16, 128], FL4)
    xb = gl.allocate_shared_memory(gl.uint8, [2, 8, 16, 128], FL4)
    xra = xa._reinterpret(gl.uint8, [4, 8, 16, 128], FL3)
    xrb = xb._reinterpret(gl.uint8, [2, 8, 16, 128], FL3)
    for i in gl.static_range(6):
        if i < 4:
            xra.index(i).store(gl.load(x_ptr + n4 * 6144 + (6 * w4 + i) * 128 + b4))
        else:
            xrb.index(i - 4).store(gl.load(x_ptr + n4 * 6144 + (6 * w4 + i) * 128 + b4))
    gl.barrier()
    if I < 4:
        b = xra.index(I).permute((0, 2, 1)).load(B3)
    else:
        b = xrb.index(I - 4).permute((0, 2, 1)).load(B3)
    ws = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, SL3)))[:, None, None]
    rs = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, SL3)))[None, :, None]
    gs = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, SL3)))[None, None, :]
    sb = gl.load(xs_ptr + rs * 192 + (6 * ws + I) * 4 + gs)  # [16 columns, 48 chunks, 4] codes
    w = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, A3)))[:, None, None]
    r = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, A3)))[None, :, None]
    kk = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, A3)))[None, None, :]
    a = gl.load(a_ptr + w * 1024 + r * 64 + kk)
    sa = gl.load(sa_ptr + ws * 64 + rs * 4 + gs)
    acc = cdna4.mfma_scaled(a, sa, "e2m1", b.to(gl.float8e4nv, bitcast=True), sb, "e4m3",
                            gl.zeros([8, 16, 16], gl.float32, MF3))
    OL: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0])
    accb = gl.convert_layout(acc, OL)
    w3 = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, OL)))[:, None, None]
    r3 = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, OL)))[None, :, None]
    n3 = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, OL)))[None, None, :]
    gl.store(c_ptr + w3 * 256 + r3 * 16 + n3, accb)


FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
g = torch.Generator().manual_seed(1)


def deq_a(a, sa):
    af = torch.stack([FP4[(a & 15).long()], FP4[(a >> 4).long()]], -1).reshape(8, 16, 128)
    return af * (2.0 ** (sa.float() - 127)).repeat_interleave(32, -1)


def fp8(x):
    return x.view(torch.float8_e4m3fn).float()


a = torch.randint(0, 256, (8, 16, 64), generator=g, dtype=torch.uint8)
sa = torch.randint(122, 132, (8, 16, 4), generator=g, dtype=torch.uint8)
# down: 41 rows x 6 chunks of 128 B, codes 4 a line
NROW, NL = 41, 256
mid = (torch.randint(0, 6, (NL, 128), generator=g) * 8 + 32).to(torch.uint8)
msc = torch.randint(122, 132, (NL, 4), generator=g, dtype=torch.uint8)
rows = torch.randint(0, NROW, (16,), generator=g, dtype=torch.int32)
c = torch.zeros(8, 16, 16, device="cuda")
k_down[(1,)](a.cuda(), sa.cuda(), mid.view(torch.int64).cuda(), msc.cuda(), rows.cuda(), c, NL, num_warps=8)
ref = torch.zeros(8, 16, 16)
for w in range(8):
    lines = rows.long() * 6 + w % 6
    bf = fp8(mid[lines]).T * (2.0 ** (msc[lines].float() - 127)).T.repeat_interleave(32, 0)  # [128, 16]
    ref[w] = deq_a(a, sa)[w] @ bf
print("down max err", (c.cpu() - ref).abs().max().item(), "ref max", ref.abs().max().item())
# up / gate
x = (torch.randint(0, 6, (16, 6144), generator=g) * 8 + 32).to(torch.uint8)
xsc = torch.randint(122, 132, (16, 48, 4), generator=g, dtype=torch.uint8)
for I in (0, 5):
    c = torch.zeros(8, 16, 16, device="cuda")
    k_ug[(1,)](a.cuda(), sa.cuda(), x.cuda(), xsc.cuda(), c, I, num_warps=8)
    ref = torch.zeros(8, 16, 16)
    for w in range(8):
        ch = 6 * w + I
        bf = fp8(x[:, ch * 128:(ch + 1) * 128]).T * (2.0 ** (xsc[:, ch].float() - 127)).T.repeat_interleave(32, 0)
        ref[w] = deq_a(a, sa)[w] @ bf
    print(f"ug round {I} max err", (c.cpu() - ref).abs().max().item(), "ref max", ref.abs().max().item())
