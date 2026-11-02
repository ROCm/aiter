from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def post_body(R,X,M,Out,B:gl.constexpr=128,SHUFFLED:gl.constexpr=False,FMA:gl.constexpr=2):
 pid=gl.program_id(0)
 if pid<4096//B:
  SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[4,B],[1,0])
  XL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,B],[1,0])
  rb=gl.allocate_shared_memory(gl.bfloat16,[4,B],SL);xb=gl.allocate_shared_memory(gl.bfloat16,[1,B],XL)
  rd=gl.amd.gfx1250.tdm.make_tensor_descriptor(R,[4,4096],[4096,1],[4,B],SL)
  xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(X,[1,4096],[4096,1],[1,B],XL)
  gl.amd.gfx1250.tdm.async_load(rd,[0,pid*B],rb,cache_modifier='.cg')
  gl.amd.gfx1250.tdm.async_load(xd,[0,pid*B],xb,cache_modifier='.cg')
  gl.amd.gfx1250.tdm.async_wait(0)
  L:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])
  r=rb.load(L).to(gl.float32);x=gl.sum(xb.load(L).to(gl.float32),0)
  h=gl.arange(0,4,layout=gl.SliceLayout(1,L));k=pid*B+gl.arange(0,B,layout=gl.SliceLayout(0,L))
  for j in gl.static_range(4):
   y=x*gl.load(M+4+j)
   for i in gl.static_range(4):
    ri=gl.sum(gl.where(h[:,None]==i,r,0),0);c=gl.load(M+8+i*4+j)
    if FMA==2 and i==0:y=gl.fma(x,gl.load(M+4+j),ri*c)
    elif FMA:y=gl.fma(ri,c,y)
    else:y=y+ri*c
   if SHUFFLED:off=(k//64)*256+j*64+k%64
   else:off=j*4096+k
   y=gl.inline_asm_elementwise("v_mov_b32 $0, $1",constraints="=v,v",args=[y],dtype=gl.float32,is_pure=True,pack=1)
   gl.store(Out+off,y.to(gl.bfloat16))
@g.jit
def post(R,X,M,Out,B:gl.constexpr=128,SHUFFLED:gl.constexpr=False,FMA:gl.constexpr=2):
 post_body(R,X,M,Out,B,SHUFFLED,FMA)
