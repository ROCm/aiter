import os
from triton.experimental.gluon import language as gl
MHC_STAGES=gl.constexpr(int('2'))
from previous.edge_sync import edge,publish,consume,prefetch_bytes
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from previous.mhc_native_inline import _mhc_post_pre_gemm_sqrsum_gfx1250_kernel as project,_mhc_pre_reduce_apply_gfx1250_kernel as reduce_ws
from combined_scaled import sync
@g.jit
def post_pre(X,R,Mix,NR,NewMix,Y,S,B,NW,Args,H,M:gl.constexpr,L:gl.constexpr,V:gl.constexpr,Edges=None,Epoch=0,InputTiles=None):
 p=gl.program_id(0)
 W,G,SQ,PO,CO,CT=Args
 if p<64:
  if InputTiles is not None:consume(InputTiles+(p//2)*64,H,M,True,1)
  project(G,SQ,NR,X,R,W,Mix+4,Mix+8,M,4096,32,4096,4,24,32,16,64,1,64,MHC_STAGES,False,True,*L,PIDM=0,PIDK=p,COEF_STRIDE=24)
 if Edges is not None:edge(Edges,H,64,p<64,p<M,Epoch=Epoch)
 elif p<64:sync(CT,H,64,3)
 if p<M:
  reduce_ws(PO,CO,Y,G,SQ,S,B,NR,NW,M,32,1.e-6,1.e-6,1.e-6,1.e-6,2.,4096,4,64,1,4096,V[0],4,20,False,True,V[1],V[2],V[3],V[4],V[5],V[6],RES_KS=64)
  # reduction returns only after its worker warps publish post/comb.
  i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]))
  v=gl.load(CO+p*16+i-8,(i>=8)&(i<24),0.)
  post=gl.load(PO+p*4+i-4,(i>=4)&(i<8),0.)
  gl.store(NewMix+p*24+i,gl.where(i<8,post,v),(i>=4)&(i<24))
