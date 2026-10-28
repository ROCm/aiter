"""Batched scaled MFMA with warps [4 batches, 2 M halves, 1]: A [4, 32, 128], B [4, 128, 16] (the M
halves share B), acc [4, 32, 16]. Which warp order do the explicit scale layouts need?"""
import sys
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4

MF4 = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[4, 2, 1]))
A4 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MF4, k_width=16))
B4 = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MF4, k_width=16))
LANES = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]]
# order 0: wave = 2 b + m ; order 1: wave = b + 4 m
WA = [[[0, 16, 0], [1, 0, 0], [2, 0, 0]], [[1, 0, 0], [2, 0, 0], [0, 16, 0]]]
WB = [[[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[1, 0, 0], [2, 0, 0], [0, 0, 0]]]
o = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SA = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[], lane_bases=LANES, warp_bases=WA[o], block_bases=[],
                                             shape=[4, 32, 4]))
SB = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[], lane_bases=LANES, warp_bases=WB[o], block_bases=[],
                                             shape=[4, 16, 4]))


@gluon.jit
def k(a_ptr, b_ptr, c_ptr):
    BLA: gl.constexpr = gl.BlockedLayout([1, 1, 16], [1, 8, 8], [4, 2, 1], [2, 1, 0])
    BLB: gl.constexpr = gl.BlockedLayout([1, 16, 1], [1, 4, 16], [4, 2, 1], [1, 2, 0])
    w = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, BLA)))[:, None, None]
    r = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, BLA)))[None, :, None]
    kk = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(1, BLA)))[None, None, :]
    a = gl.load(a_ptr + w * 4096 + r * 128 + kk)
    wb = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, BLB)))[:, None, None]
    kb = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(2, BLB)))[None, :, None]
    nb = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, BLB)))[None, None, :]
    b = gl.load(b_ptr + wb * 2048 + kb * 16 + nb)
    a = gl.convert_layout(a, A4)
    b = gl.convert_layout(b, B4)
    sa = gl.full([4, 32, 4], 127, gl.uint8, SA)
    sb = gl.full([4, 16, 4], 127, gl.uint8, SB)
    acc = gl.zeros([4, 32, 16], gl.float32, MF4)
    acc = cdna4.mfma_scaled(a.to(gl.float8e4nv, bitcast=True), sa, "e4m3", b.to(gl.float8e4nv, bitcast=True), sb,
                            "e4m3", acc)
    OL: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 4, 16], [4, 2, 1], [2, 1, 0])
    accb = gl.convert_layout(acc, OL)
    w3 = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, OL)))[:, None, None]
    r3 = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, OL)))[None, :, None]
    n3 = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, OL)))[None, None, :]
    gl.store(c_ptr + w3 * 512 + r3 * 16 + n3, accb)


a = (torch.randint(0, 3, (4, 32, 128), device="cuda") * 8 + 48).to(torch.uint8)
b = (torch.randint(0, 3, (4, 128, 16), device="cuda") * 8 + 48).to(torch.uint8)
c = torch.zeros(4, 32, 16, device="cuda")
k[(1,)](a, b, c, num_warps=8)
ref = torch.einsum("wrk,wkn->wrn", a.view(torch.float8_e4m3fn).float(), b.view(torch.float8_e4m3fn).float())
print("order", o, "max err", (c - ref).abs().max().item(), "ref max", ref.abs().max().item())
