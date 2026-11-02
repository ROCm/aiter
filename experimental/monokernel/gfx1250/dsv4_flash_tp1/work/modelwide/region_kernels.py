from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import complete,sync
from output_scaled import output_body
from mhc_precise import candidate_body as mhc_body
from post_precise import post_body
from layout import unshuffle_body
from ffn_extended import ffn_body

@g.jit
def qkv(A):
 complete(*A[10:31],BN=128,GRID=256,STYLE=3,U=4,QAMODE=0,ALLTDM=True,EARLY=True,PREFETCH=2)

@g.jit
def output(A):
 output_body(A[38],A[21],A[22],*A[40:],A[29],A[30])

@g.jit
def mhc(R,Plain,PostR,Prev,MPrev,R32,W,S,B,N,M,Temp,Raw,X,Counter,Health,HAS_PREV:gl.constexpr):
 unshuffle_body(R,Plain)
 sync(Counter,Health,256,3)
 if HAS_PREV:
  post_body(Plain,Prev,MPrev,PostR,128,False,2,FP32=R32)
  sync(Counter,Health,256,3)
  mhc_body(PostR,W,S,B,N,M,Temp,Raw,X,Counter,Health,FP32=R32)
 else:
  mhc_body(Plain,W,S,B,N,M,Temp,Raw,X,Counter,Health)
