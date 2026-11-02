import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def probe(P):
 ml:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,32])
 gl.static_print('OUTPUT',gl.to_linear_layout(ml,[16,128]))
 gl.static_print('A',gl.to_linear_layout(gl.DotOperandLayout(0,ml,8),[16,128]))
 gl.static_print('B',gl.to_linear_layout(gl.DotOperandLayout(1,ml,8),[128,128]))
 gl.store(P,0.)
probe[(1,)](torch.empty(1,device='cuda'),num_warps=4)
