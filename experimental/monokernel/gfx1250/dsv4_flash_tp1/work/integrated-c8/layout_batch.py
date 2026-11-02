from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def unshuffle(X,Y,M:gl.constexpr):
 p=gl.program_id(0)
 for task in range(p,M*64,256):
  row=task//64;tile=task%64
  SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[4,64],[1,0])
  LL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
  sm=gl.allocate_shared_memory(gl.bfloat16,[4,64],SL)
  desc=gl.amd.gfx1250.tdm.make_tensor_descriptor(X+tile*256*M+row*64,[4,64],[M*64,1],[4,64],SL)
  gl.amd.gfx1250.tdm.async_load(desc,[0,0],sm);gl.amd.gfx1250.tdm.async_wait(0)
  h=gl.arange(0,4,layout=gl.SliceLayout(1,LL));k=gl.arange(0,64,layout=gl.SliceLayout(0,LL))
  gl.store(Y+row*16384+h[:,None]*4096+tile*64+k[None,:],sm.load(LL))
@g.jit
def shuffle(X,Y,M:gl.constexpr):
 p=gl.program_id(0)
 for task in range(p,M*64,256):
  row=task//64;tile=task%64
  SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[4,64],[1,0])
  LL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
  sm=gl.allocate_shared_memory(gl.bfloat16,[4,64],SL)
  desc=gl.amd.gfx1250.tdm.make_tensor_descriptor(X+row*16384,[4,4096],[4096,1],[4,64],SL)
  gl.amd.gfx1250.tdm.async_load(desc,[0,tile*64],sm);gl.amd.gfx1250.tdm.async_wait(0)
  h=gl.arange(0,4,layout=gl.SliceLayout(1,LL));k=gl.arange(0,64,layout=gl.SliceLayout(0,LL))
  gl.store(Y+tile*256*M+h[:,None]*M*64+row*64+k[None,:],sm.load(LL))
