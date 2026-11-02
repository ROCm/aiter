from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from layer_kernel import full_layer
from combined import sync
@g.jit
def serving_full_layer(A,F,Positions,CosTable,SinTable,RealCache,RealIndices,RealPtr,Dest,
 W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,REALPAGES:gl.constexpr):
 p=gl.program_id(0);n=gl.load(RealPtr+1)-gl.load(RealPtr)
 if p<128:
  SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,512],[1,0])
  b=gl.allocate_shared_memory(gl.bfloat16,[1,512],SL)
  slot=gl.load(RealIndices+p,p<n,0)
  d=gl.amd.gfx1250.tdm.make_tensor_descriptor(RealCache+gl.maximum(slot,0).to(gl.int64)*512,[1,512],[512,1],[1,512],SL)
  gl.amd.gfx1250.tdm.async_load(d,[0,0],b);gl.amd.gfx1250.tdm.async_wait(0)
  L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0]);i=gl.arange(0,512,layout=L)
  v=gl.convert_layout(gl.sum(b.load(gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])).to(gl.float32),0),L)
  gl.store(A[31]+p*512+i,v)
  gl.store(A[32]+p,gl.where(p==n-1,127,p))
 if p==0:
  i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]));pos=gl.load(Positions)
  gl.store(A[21]+i,gl.load(CosTable+pos*32+i));gl.store(A[22]+i,gl.load(SinTable+pos*32+i))
  gl.store(A[33],0);gl.store(A[33]+1,n)
 sync(A[29],A[30],256,3)
 full_layer(A,F,W1S,S1S,W2S,S2S,True)
 if p==0:
  i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]));dest=gl.load(Dest).to(gl.int64)
  gl.store(RealCache+dest*512+i,gl.load(A[23]+i),dest>=0)
