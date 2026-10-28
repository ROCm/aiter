"""Batched (a batch per wave) scaled MFMA: e2m1 A x e4m3 B, random E8M0 scales, vs torch."""
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4

MF3 = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[8, 1, 1]))
A3 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MF3, k_width=16))
B3 = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MF3, k_width=16))
SL3 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[], lane_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]],
                                              warp_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0]], block_bases=[], shape=[8, 16, 4]))


@gluon.jit
def k(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr):
    w = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, A3)))[:, None, None]
    r = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, A3)))[None, :, None]
    kk = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, A3)))[None, None, :]
    a = gl.load(a_ptr + w * 1024 + r * 64 + kk)  # packed fp4, 2 a byte
    wb = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, B3)))[:, None, None]
    kb = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(2, B3)))[None, :, None]
    nb = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, B3)))[None, None, :]
    b = gl.load(b_ptr + wb * 2048 + kb * 16 + nb)
    ws = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, SL3)))[:, None, None]
    rs = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, SL3)))[None, :, None]
    gs = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, SL3)))[None, None, :]
    sa = gl.load(sa_ptr + ws * 64 + rs * 4 + gs)
    sb = gl.load(sb_ptr + ws * 64 + rs * 4 + gs)
    acc = gl.zeros([8, 16, 16], gl.float32, MF3)
    acc = cdna4.mfma_scaled(a, sa, "e2m1", b.to(gl.float8e4nv, bitcast=True), sb, "e4m3", acc)
    OL: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0])
    accb = gl.convert_layout(acc, OL)
    w3 = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, OL)))[:, None, None]
    r3 = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, OL)))[None, :, None]
    n3 = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, OL)))[None, None, :]
    gl.store(c_ptr + w3 * 256 + r3 * 16 + n3, accb)


FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
g = torch.Generator().manual_seed(0)
a = torch.randint(0, 256, (8, 16, 64), generator=g, dtype=torch.uint8)
b = (torch.randint(0, 5, (8, 128, 16), generator=g) * 8 + 40).to(torch.uint8)
sa = torch.randint(122, 132, (8, 16, 4), generator=g, dtype=torch.uint8)
sb = torch.randint(122, 132, (8, 16, 4), generator=g, dtype=torch.uint8)
c = torch.zeros(8, 16, 16, device="cuda")
k[(1,)](a.cuda(), b.cuda(), sa.cuda(), sb.cuda(), c, num_warps=8)
af = torch.stack([FP4[(a & 15).long()], FP4[(a >> 4).long()]], -1).reshape(8, 16, 128)
af = af * (2.0 ** (sa.float() - 127)).repeat_interleave(32, -1)
bf = b.view(torch.float8_e4m3fn).float() * (2.0 ** (sb.float() - 127)).permute(0, 2, 1).repeat_interleave(32, 1)
ref = torch.einsum("wrk,wkn->wrn", af, bf)
print("max err", (c.cpu() - ref).abs().max().item(), "ref max", ref.abs().max().item())
