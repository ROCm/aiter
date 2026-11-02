from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from mhc_native_inline import _mhc_post_pre_gemm_sqrsum_gfx1250_kernel as project, _reduce_apply_main as apply, _RAConfig, _mhc_pre_reduce_apply_gfx1250_kernel as reduce_ws
from combined_scaled import sync
@g.jit
def region(X,R,W,P,C,S,B,NW,G,SQ,NR,PO,CO,Y,CT,H,L:gl.constexpr,V:gl.constexpr):
 pid=gl.program_id(0)
 if pid<64:
  project(G,SQ,NR,X,R,W,P,C,8,4096,32,4096,4,24,32,16,64,1,64,2,True,True,*L,PIDM=0,PIDK=pid)
 if pid<64:sync(CT,H,64,3)
 if pid<8:
  reduce_ws(PO,CO,Y,G,SQ,S,B,NR,NW,8,32,1.e-6,1.e-6,1.e-6,1.e-6,2.,4096,4,64,1,4096,V[0],4,20,True,True,V[1],V[2],V[3],V[4],V[5],V[6],RES_KS=64)
