"""Does Gluon's cdna4.mfma_scaled take a batched (3D, a batch per wave) operand pair?"""
import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4

MF3 = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[8, 1, 1]))
A3 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MF3, k_width=16))
B3 = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MF3, k_width=16))


@gluon.jit
def k(a_ptr, b_ptr, c_ptr):
    # a [8, 16, 128] fp8, b [8, 128, 16] fp8 -> c [8, 16, 16] f32
    BLA: gl.constexpr = gl.BlockedLayout([1, 1, 16], [1, 8, 8], [8, 1, 1], [2, 1, 0])
    BLB: gl.constexpr = gl.BlockedLayout([1, 16, 1], [1, 4, 16], [8, 1, 1], [1, 2, 0])
    w = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, BLA)))[:, None, None]
    r = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, BLA)))[None, :, None]
    kk = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(1, BLA)))[None, None, :]
    a = gl.load(a_ptr + w * 2048 + r * 128 + kk)
    wb = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, BLB)))[:, None, None]
    kb = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(2, BLB)))[None, :, None]
    nb = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, BLB)))[None, None, :]
    b = gl.load(b_ptr + wb * 2048 + kb * 16 + nb)
    a = gl.convert_layout(a, A3)
    b = gl.convert_layout(b, B3)
    SA: gl.constexpr = cdna4.get_mfma_scale_layout(A3, [8, 16, 4])
    SB: gl.constexpr = cdna4.get_mfma_scale_layout(B3, [8, 16, 4])
    sa = gl.full([8, 16, 4], 127, gl.uint8, SA)
    sb = gl.full([8, 16, 4], 127, gl.uint8, SB)
    acc = gl.zeros([8, 16, 16], gl.float32, MF3)
    acc = cdna4.mfma_scaled(a.to(gl.float8e4nv, bitcast=True), sa, "e4m3", b.to(gl.float8e4nv, bitcast=True), sb,
                            "e4m3", acc)
    tot = gl.sum(acc, axis=0)  # the waves' K slices summed (cross-warp)
    o = gl.arange(0, 16, layout=gl.SliceLayout(1, tot.type.layout))[:, None] * 16 + \
        gl.arange(0, 16, layout=gl.SliceLayout(0, tot.type.layout))[None, :]
    gl.store(c_ptr + o, tot)


a = (torch.randint(0, 3, (8, 16, 128), device="cuda") * 8 + 48).to(torch.uint8)  # small fp8 values
b = (torch.randint(0, 3, (8, 128, 16), device="cuda") * 8 + 48).to(torch.uint8)
c = torch.zeros(16, 16, device="cuda")
k[(1,)](a, b, c, num_warps=8)
af = a.view(torch.float8_e4m3fn).float()
bf = b.view(torch.float8_e4m3fn).float()
ref = torch.einsum("wrk,wkn->rn", af, bf)
print("max err", (c - ref).abs().max().item(), "ref max", ref.abs().max().item())
