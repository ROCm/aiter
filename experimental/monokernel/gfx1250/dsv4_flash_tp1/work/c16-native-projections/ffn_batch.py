import os
from triton.experimental.gluon import language as gl
SHARED_TUNE = gl.constexpr(int(os.getenv('SHARED_TUNE','1')))
SHARED_BK = gl.constexpr(int(os.getenv('SHARED_BK','512')))
SHARED_NB = gl.constexpr(int(os.getenv('SHARED_NB','4')))
from edge_sync import edge,publish,consume,prefetch_bytes
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined import sync
from mhc_batch import mhc_batch as mhc_body
from post_batch import post_batch as post_body
from optimized_moe import fused as moe_body
from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_mxfp8 import _gemm_mxfp8_preshuffle_bandwidth_bound_kernel as _shared_gemm
shared_gemm = _shared_gemm.fn
from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import _gemm_a16w16_bandwidth_bound_kernel as router_gemm

@g.jit
def quant_body(X,Q,S,N:gl.constexpr,GROUP:gl.constexpr,STRIDE:gl.constexpr=1,M:gl.constexpr=1):
 p=gl.program_id(0)
 for task in range(p,M*(N//GROUP),256):
  row=task//(N//GROUP);pcol=task%(N//GROUP)
  SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[32,8]],[1,GROUP],[1,0])
  sm=gl.allocate_shared_memory(gl.bfloat16,[1,GROUP],SL)
  desc=gl.amd.gfx1250.tdm.make_tensor_descriptor(X+row*N,[1,N],[N,1],[1,GROUP],SL)
  gl.amd.gfx1250.tdm.async_load(desc,[0,pcol*GROUP],sm);gl.amd.gfx1250.tdm.async_wait(0)
  L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
  x=gl.convert_layout(gl.sum(sm.load(gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])).to(gl.float32),0),L)
  v=gl.max(gl.abs(x),0)/448.
  e=((v.to(gl.uint32,bitcast=True)+0x7fffff)&0x7f800000)
  scale=e.to(gl.float32,bitcast=True)
  q=x*gl.where(scale==0,0.,1./scale)
  i=gl.arange(0,GROUP,layout=L)
  gl.store(Q+row*N+pcol*GROUP+i,q.to(gl.float8e4nv));gl.store(S+(row*(N//GROUP)+pcol if STRIDE==1 else pcol*STRIDE+row),(e>>23).to(gl.uint8))

@g.jit
def router_body(X,W,Y,M:gl.constexpr):
 if gl.program_id(0)<16:
  SA:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[512,8]],[16,512],[1,0])
  SB:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[512,8]],[16,512],[1,0])
  ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=((0,1),(1,0)),instr_shape=[16,16,32])
  router_gemm(X,W,Y,None,M,256,4096,4096,1,1,4096,256,1,16,16,512,4,'TN',SA,SB,ML,gl.DotOperandLayout(0,ML,8),gl.DotOperandLayout(1,ML,8),'',False,False)

from optimized_routing import route_body

@g.jit
def shared_body(X,XS,W,WS,Y,K:gl.constexpr,BN:gl.constexpr,M:gl.constexpr):
 if SHARED_TUNE:
  if gl.program_id(0)<128:
   shared_gemm(X,W,Y,XS,WS,M,4096,K,K,1,K*16,1,0,4096,1,1,max(8,M),K//128,1,
    16,32,SHARED_BK,1,128,True,1,K,True,4,((0,1),(0,2)),None,min(SHARED_NB,K//SHARED_BK),1,True,1,1,1,1)
  return
 if gl.program_id(0)<4096//BN:
  shared_gemm(X,W,Y,XS,WS,M,4096,K,K,1,K*16,1,0,4096,1,1,max(8,M),K//128,1,
   32,BN,256,1,128,True,1,K,True,4,((0,1),(1,0)),None,3,1,True,1,1,1,1)

@g.jit
def activation_quant(GU,A,AQ,AS,M:gl.constexpr):
 p=gl.program_id(0)
 if p<M*16:
  row=p//16;p=p%16
  SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,128],[1,0])
  sm=gl.allocate_shared_memory(gl.bfloat16,[2,1,128],SL)
  d=gl.amd.gfx1250.tdm.make_tensor_descriptor(GU+row*4096,[1,4096],[4096,1],[1,128],SL)
  gl.amd.gfx1250.tdm.async_load(d,[0,p*128],sm.index(0));gl.amd.gfx1250.tdm.async_load(d,[0,2048+p*128],sm.index(1));gl.amd.gfx1250.tdm.async_wait(0)
  LL:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,4],[1,0]);L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
  gate=gl.convert_layout(gl.sum(sm.index(0).load(LL).to(gl.float32),0),L);up=gl.convert_layout(gl.sum(sm.index(1).load(LL).to(gl.float32),0),L)
  gate=gl.minimum(gate,10.);up=gl.minimum(gl.maximum(up,-10.),10.)
  v=(gate/(1.+gl.exp(-gate))*up).to(gl.bfloat16).to(gl.float32)
  bits=gl.maximum(gl.max(gl.abs(v),0),1.e-10)/448.;e=((bits.to(gl.uint32,bitcast=True)+0x7fffff)&0x7f800000)
  q=v/e.to(gl.float32,bitcast=True);i=gl.arange(0,128,layout=L)
  gl.store(A+row*2048+p*128+i,v);gl.store(AQ+row*2048+p*128+i,q);gl.store(AS+p*max(8,M)+row,(e>>23).to(gl.uint8))

@g.jit
def ffn_body(R,Out,HFN,HFS,HFB,FN,Mix,Temp,Raw,X,XQ,XS,MX,MS,GU,ACT,AQ,AS,SO,Gate,Logits,
 SW1,SS1,SW2,SS2,W1,S1,W2,S2,I,IS,Y,Hist,Offs,Pad,Map,Gather,Scatter,Gamma,Count,Z,Sum,Counter,Health,
 W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,M:gl.constexpr,Chosen,Weights,DO_PRE:gl.constexpr=True,POST_SHUFFLED:gl.constexpr=False,Bias=None,HAS_BIAS:gl.constexpr=False,FP32IN=None,FP32OUT=None,Edges=None,OVERLAP:gl.constexpr=False,Epoch=0,Tiles=None,TILE_MASK:gl.constexpr=0):
 gl.static_assert(not SHARED_TUNE or Edges is None, "retile shared-expert mailboxes before enabling edge mode")
 if Tiles is not None and TILE_MASK&8:gl.static_assert(Edges is not None)
 if DO_PRE:
  mhc_body(R,HFN,HFS,HFB,FN,Mix,Temp,Raw,X,Counter,Health,M,FP32=FP32IN)
  sync(Counter,Health,256,3)
 if Edges is not None:
  p=gl.program_id(0)
  if OVERLAP & 1:
   prefetch_bytes(Gate,256*4096*2)
   prefetch_bytes(SW1,4096*4096)
   prefetch_bytes(SS1,4096*4096//128//128)
   if HAS_BIAS:prefetch_bytes(Bias,256*2)
  quant_body(X,XQ,XS,4096,128,max(8,M),M)
  publish(Edges+10*320,p<gl.minimum(256,M*32))
  quant_body(X,MX,MS,4096,32,1,M)
  publish(Edges+11*320,p<gl.minimum(256,M*128))
  router_body(X,Gate,Logits,M)
  edge(Edges+12*320,Health,16,p<16,p<M,Epoch=Epoch)
  route_body(Logits,Hist,Offs,Pad,Map,Gather,Scatter,Gamma,Bias,HAS_BIAS,Chosen,Weights,Counter,Health,M,Edges+13*320,Epoch=Epoch)
  publish(Edges+14*320,p<M)
  consume(Edges+10*320,Health,min(256,M*32),p<64,Epoch=Epoch)
  shared_body(XQ,XS,SW1,SS1,GU,4096,64,M)
  if Tiles is not None and TILE_MASK&8:publish(Tiles+(200+(p%32)//2)*64,p<64)
  else:publish(Edges+15*320,p<64)
  if OVERLAP & 1:
   prefetch_bytes(SW2,4096*2048)
   prefetch_bytes(SS2,4096*2048//128//128)
  if Tiles is not None and TILE_MASK&8:consume(Tiles+(200+p%16)*64,Health,4,p<M*16,1)
  else:consume(Edges+15*320,Health,64,p<M*16,Epoch=Epoch)
  activation_quant(GU,ACT,AQ,AS,M)
  edge(Edges+16*320,Health,M*16,p<M*16,p<128,Epoch=Epoch)
  shared_body(AQ,AS,SW2,SS2,SO,2048,32,M)
  if Tiles is not None and TILE_MASK&8:publish(Tiles+(216+p//4)*64,p<128)
  consume(Edges+11*320,Health,min(256,M*128),True,Epoch=Epoch)
  consume(Edges+14*320,Health,M,True,Epoch=Epoch)
  moe_body(MX,MS,W1,S1,W2,S2,I,IS,Y,Gather,Hist,Offs,Pad,Map,Gamma,Count,Health,Z,Scatter,W1S,S1S,W2S,S2S,10.,M,256,3,Tiles if TILE_MASK&4 else None)
 else:
  quant_body(X,XQ,XS,4096,128,max(8,M),M)
  quant_body(X,MX,MS,4096,32,1,M)
  router_body(X,Gate,Logits,M)
  sync(Counter,Health,256,3)
  route_body(Logits,Hist,Offs,Pad,Map,Gather,Scatter,Gamma,Bias,HAS_BIAS,Chosen,Weights,Counter,Health,M)
  shared_body(XQ,XS,SW1,SS1,GU,4096,64,M)
  sync(Counter,Health,256,3)
  activation_quant(GU,ACT,AQ,AS,M)
  sync(Counter,Health,256,3)
  shared_body(AQ,AS,SW2,SS2,SO,2048,32,M)
  moe_body(MX,MS,W1,S1,W2,S2,I,IS,Y,Gather,Hist,Offs,Pad,Map,Gamma,Count,Health,Z,Scatter,W1S,S1S,W2S,S2S,10.,M,256,3,Tiles if TILE_MASK&4 else None)
 if Tiles is None or not TILE_MASK&8:sync(Counter,Health,256,3)
 for task in range(gl.program_id(0),M*32,256):
  if Tiles is not None and TILE_MASK&8:consume(Tiles+(216+gl.program_id(0)%32)*64,Health,4,True,1)
  i=task*128+gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
  gl.store(Sum+i,(gl.load(SO+i).to(gl.float32)+gl.load(Z+i).to(gl.float32)).to(gl.bfloat16))
 if Tiles is None or not TILE_MASK&8:sync(Counter,Health,256,3)
 post_body(R,Sum,Mix,Out,M,POST_SHUFFLED,FP32OUT)
