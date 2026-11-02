from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync
from previous.mhc_pre_aiter import pre
from previous.mhc_integrated import post_pre
from previous.post_batch import post_batch
from previous.output_batch import output_body
from previous.optimized_moe import fused
from previous.ffn_batch import quant_body,router_body,shared_body,activation_quant
@g.jit
def mhc_pre(RS,R,Mix,Y,S,B,NW,Args,H,M:gl.constexpr,L:gl.constexpr,V:gl.constexpr):
 pre(R,Mix,Y,S,B,NW,Args,H,M,L,V,Shuffled=RS)
@g.jit
def mhc_mid(X,R,Mix,NR,NMix,Y,S,B,NW,Args,H,M:gl.constexpr,L:gl.constexpr,V:gl.constexpr):
 post_pre(X,R,Mix,NR,NMix,Y,S,B,NW,Args,H,M,L,V)
@g.jit
def output(O,C,S,WA,SA,WB,SB,XQ,XS,AP,AY,AQ,AS,BP,Y,Count,Health,M:gl.constexpr):
 output_body(O,C,S,WA,SA,WB,SB,XQ,XS,AP,AY,AQ,AS,BP,Y,Count,Health,M,OVERLAP=2)
@g.jit
def shared(X,Q,S,G,LG,W,WS,GU,A,AQ,AS,W2,WS2,Y,COUNT,HEALTH,M:gl.constexpr):
 quant_body(X,Q,S,4096,128,max(8,M),M)
 router_body(X,G,LG,M)
 sync(COUNT,HEALTH,256,3)
 shared_body(Q,S,W,WS,GU,4096,64,M)
 sync(COUNT,HEALTH,256,3)
 activation_quant(GU,A,AQ,AS,M)
 sync(COUNT,HEALTH,256,3)
 shared_body(AQ,AS,W2,WS2,Y,2048,32,M)
@g.jit
def moe_region(X,XQ,XS,A,Counter,Health):
 quant_body(X,XQ,XS,4096,32,1,A[-1])
 sync(Counter,Health,256,3)
 fused(XQ,XS,*A,GRID=256,BUFFERS=3)
@g.jit
def tail(R,A,B,Sum,Mix,Out,Count,Health,M:gl.constexpr):
 for task in range(gl.program_id(0),M*32,256):
  i=task*128+gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
  gl.store(Sum+i,(gl.load(A+i).to(gl.float32)+gl.load(B+i).to(gl.float32)).to(gl.bfloat16))
 sync(Count,Health,256,3)
 post_batch(R,Sum,Mix,Out,M,True)
from previous.qkv_candidate import qkv_batch
from previous.ffn_batch import ffn_body
@g.jit
def qkv_cache(A,M:gl.constexpr,Cache,Dest,Pages):
 qkv_batch(*A,M,PREFETCH_STYLE=1)
 p=gl.program_id(0)
 if p<M:
  i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]));dest=gl.load(Dest+p).to(gl.int64)
  gl.store(Cache+dest*512+i,gl.load(A[12]+p*512+i),(dest>=0)&(dest<Pages))
@g.jit
def whole_ffn(F,W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,M:gl.constexpr,Chosen,Weights):
 ffn_body(*F,W1S,S1S,W2S,S2S,M,Chosen,Weights,DO_PRE=False,POST_SHUFFLED=True)
