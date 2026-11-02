from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync
from post_batch import post_batch
@g.jit
def combine(R,SO,Z,Sum,Mix,Out,Counter,Health,FP32):
 p=gl.program_id(0)
 i=p*128+gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
 gl.store(Sum+i,(gl.load(SO+i).to(gl.float32)+gl.load(Z+i).to(gl.float32)).to(gl.bfloat16))
 sync(Counter,Health,256,3)
 post_batch(R,Sum,Mix,Out,8,False,FP32)
from mhc_batch import mhc_batch
@g.jit
def post_pre(R,Y,Mix,Mid,Mid32,W,S,B,NW,Mix2,Temp,Raw,X,CT,H):
 mhc_batch(Mid,W,S,B,NW,Mix2,Temp,Raw,X,CT,H,8,FP32=Mid32,PrevR=R,PrevY=Y,PrevMix=Mix)
