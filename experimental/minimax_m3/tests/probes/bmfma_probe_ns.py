"""Does a batched (rank-3) scaled MFMA compile on gfx950 (one 16x16x128 MFMA a wave)?"""
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4

M3 = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[4, 2, 1]))
A3 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=M3, k_width=16))
B3 = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=M3, k_width=16))
BL3 = gl.constexpr(gl.BlockedLayout([1, 1, 16], [1, 4, 16], [4, 2, 1], [2, 1, 0]))
M2 = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[2, 1]))
A2 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=M2, k_width=16))
B2 = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=M2, k_width=16))
BL2 = gl.constexpr(gl.BlockedLayout([1, 16], [4, 16], [2, 1], [1, 0]))


def i3(n0, n1, n2, L):
    a = gl.arange(0, n0, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))[:, None, None]
    b = gl.arange(0, n1, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))[None, :, None]
    c = gl.arange(0, n2, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))[None, None, :]
    return a, b, c


@gluon.jit
def k3(a_ptr, as_ptr, b_ptr, bs_ptr, out):
    q = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, BL3)))[:, None, None]
    m = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, BL3)))[None, :, None]
    kk = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, BL3)))[None, None, :]
    a = gl.convert_layout(gl.load(a_ptr + q * 2048 + m * 64 + kk), A3)
    kb = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(2, BL3)))[None, :, None]
    n = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, BL3)))[None, None, :]
    b = gl.convert_layout(gl.load(b_ptr + q * 2048 + kb * 16 + n), B3)
    ASL: gl.constexpr = cdna4.get_mfma_scale_layout(A3, [4, 32, 4])
    BSL: gl.constexpr = cdna4.get_mfma_scale_layout(B3, [4, 16, 4])
    sq = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, ASL)))[:, None, None]
    sm = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, ASL)))[None, :, None]
    sk = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, ASL)))[None, None, :]
    asc = gl.load(as_ptr + sq * 128 + sm * 4 + sk)
    tq = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, BSL)))[:, None, None]
    tn = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, BSL)))[None, :, None]
    tk = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, BSL)))[None, None, :]
    bsc = gl.load(bs_ptr + tq * 64 + tn * 4 + tk)
    acc = cdna4.mfma_scaled(a, None, "e2m1", b.to(gl.float8e4nv, bitcast=True), None, "e4m3",
                            gl.zeros([4, 32, 16], gl.float32, M3))
    oq = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, M3)))[:, None, None]
    om = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, M3)))[None, :, None]
    on = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, M3)))[None, None, :]
    gl.store(out + oq * 512 + om * 16 + on, acc)


@gluon.jit
def k2(a_ptr, as_ptr, b_ptr, bs_ptr, out, q):
    m = gl.arange(0, 32, layout=gl.SliceLayout(1, BL2))[:, None]
    kk = gl.arange(0, 64, layout=gl.SliceLayout(0, BL2))[None, :]
    a = gl.convert_layout(gl.load(a_ptr + q * 2048 + m * 64 + kk), A2)
    kb = gl.arange(0, 128, layout=gl.SliceLayout(1, BL2))[:, None]
    n = gl.arange(0, 16, layout=gl.SliceLayout(0, BL2))[None, :]
    b = gl.convert_layout(gl.load(b_ptr + q * 2048 + kb * 16 + n), B2)
    ASL: gl.constexpr = cdna4.get_mfma_scale_layout(A2, [32, 4])
    BSL: gl.constexpr = cdna4.get_mfma_scale_layout(B2, [16, 4])
    sm = gl.arange(0, 32, layout=gl.SliceLayout(1, ASL))[:, None]
    sk = gl.arange(0, 4, layout=gl.SliceLayout(0, ASL))[None, :]
    asc = gl.load(as_ptr + q * 128 + sm * 4 + sk)
    tn = gl.arange(0, 16, layout=gl.SliceLayout(1, BSL))[:, None]
    tk = gl.arange(0, 4, layout=gl.SliceLayout(0, BSL))[None, :]
    bsc = gl.load(bs_ptr + q * 64 + tn * 4 + tk)
    acc = cdna4.mfma_scaled(a, asc, "e2m1", b.to(gl.float8e4nv, bitcast=True), bsc, "e4m3",
                            gl.zeros([32, 16], gl.float32, M2))
    om = gl.arange(0, 32, layout=gl.SliceLayout(1, M2))[:, None]
    on = gl.arange(0, 16, layout=gl.SliceLayout(0, M2))[None, :]
    gl.store(out + q * 512 + om * 16 + on, acc)


torch.manual_seed(0)
dev = "cuda"
a = torch.randint(0, 256, (4, 32, 64), dtype=torch.uint8, device=dev)
b = (torch.randn(4, 128, 16, device=dev)).to(torch.float8_e4m3fn).view(torch.uint8)
asc = torch.randint(120, 132, (4, 32, 4), dtype=torch.uint8, device=dev)
bsc = torch.randint(120, 132, (4, 16, 4), dtype=torch.uint8, device=dev)
o3 = torch.zeros(4, 32, 16, device=dev)
o2 = torch.zeros(4, 32, 16, device=dev)
cc = k3[(1,)](a, asc, b, bsc, o3, num_warps=8)
for q in range(4):
    k2[(1,)](a, asc, b, bsc, o2, q, num_warps=2)
torch.cuda.synchronize()
s = cc.asm["amdgcn"]
print("mfma in batched:", s.count("v_mfma_scale"), "max abs diff", (o3 - o2).abs().max().item(), "norm", o2.norm().item())
