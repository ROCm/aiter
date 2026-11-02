from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync
@g.jit
def stage(A,Positions,CosTable,SinTable,RealCache,RealIndices,RealPtr,M:gl.constexpr,T:gl.constexpr):
 p=gl.program_id(0)
 if p<M:
  i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]));pos=gl.load(Positions+p)
  gl.store(A[21]+p*32+i,gl.load(CosTable+pos*32+i));gl.store(A[22]+p*32+i,gl.load(SinTable+pos*32+i))
  n=gl.load(RealPtr+p+1)-gl.load(RealPtr+p)
  gl.store(A[33]+p*2,0);gl.store(A[33]+p*2+1,n)
 for task in range(p,M*T,256):
  token=task//T;row=task%T
  start=gl.load(RealPtr+token);n=gl.load(RealPtr+token+1)-start
  SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,512],[1,0])
  sm=gl.allocate_shared_memory(gl.bfloat16,[1,512],SL)
  slot=gl.load(RealIndices+start+row,row<n,0)
  desc=gl.amd.gfx1250.tdm.make_tensor_descriptor(RealCache+gl.maximum(slot,0).to(gl.int64)*512,[1,512],[512,1],[1,512],SL)
  gl.amd.gfx1250.tdm.async_load(desc,[0,0],sm);gl.amd.gfx1250.tdm.async_wait(0)
  L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0]);i=gl.arange(0,512,layout=L)
  v=gl.convert_layout(gl.sum(sm.load(gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])).to(gl.float32),0),L)
  gl.store(A[31]+task*512+i,v)
  gl.store(A[32]+task,gl.where(row==n-1,T-1,row))
 sync(A[29],A[30],256,3)
