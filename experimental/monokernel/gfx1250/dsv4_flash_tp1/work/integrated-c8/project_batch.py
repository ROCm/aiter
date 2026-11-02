from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import _gemm_a16w16_bandwidth_bound_kernel as gemm
from compressor import compressor_body
from sync import sync

@g.jit
def project_body(X,W,Y,N:gl.constexpr,ROWS:gl.constexpr):
 if gl.program_id(0)<N//16:
  S:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[512,8]],[16,512],[1,0])
  M:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=((0,1),(1,0)),instr_shape=[16,16,32])
  gemm(X,W,Y,None,ROWS,N,4096,4096,1,1,4096,N,1,16,16,512,4,'TN',S,S,M,gl.DotOperandLayout(0,M,8),gl.DotOperandLayout(1,M,8),'',False,False)

