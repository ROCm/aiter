import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4

MFMA = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[8, 1]))
A_L = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MFMA, k_width=16))
B_L = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=16))
MFMA2 = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 8]))
B2 = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MFMA2, k_width=16))
A2 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MFMA2, k_width=16))

@gluon.jit
def k(x):
    gl.static_print("A_L fp8 [128,128]", gl.to_linear_layout(A_L, [128, 128]))
    gl.static_print("B_L fp8 [128,16]", gl.to_linear_layout(B_L, [128, 16]))
    gl.static_print("A_L fp4 [128,64]", gl.to_linear_layout(A_L, [128, 64]))
    gl.static_print("B2 [128,256]", gl.to_linear_layout(B2, [128, 256]))
    gl.static_print("A2 [16,128]", gl.to_linear_layout(A2, [16, 128]))
    gl.static_print("MFMA2 acc [16,256]", gl.to_linear_layout(MFMA2, [16, 256]))
    gl.static_print("A2 P [16,256]", gl.to_linear_layout(A2, [16, 256]))
    gl.static_print("B2 V [256,128]", gl.to_linear_layout(B2, [256, 128]))
    gl.static_print("MFMA2 acc [16,128]", gl.to_linear_layout(MFMA2, [16, 128]))

x = torch.zeros(4, device="cuda")
k[(1,)](x, num_warps=8)
