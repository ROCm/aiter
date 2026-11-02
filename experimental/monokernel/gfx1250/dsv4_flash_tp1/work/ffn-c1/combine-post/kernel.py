from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def combine_post(Routed,Shared,R,Post,Comb,Out,M:gl.constexpr=1,B:gl.constexpr=128):
 pid=gl.program_id(0);row=pid//(4096//B);tile=pid%(4096//B)
 RL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[B//64,4,64],[2,1,0])
 XL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,B],[1,0])
 RLAY:gl.constexpr=gl.BlockedLayout([1,1,4],[1,1,32],[1,4,1],[2,1,0])
 L:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])
 rb=gl.allocate_shared_memory(gl.bfloat16,[B//64,4,64],RL)
 xb=gl.allocate_shared_memory(gl.bfloat16,[1,B],XL);sb=gl.allocate_shared_memory(gl.bfloat16,[1,B],XL)
 rd=gl.amd.gfx1250.tdm.make_tensor_descriptor(R+tile*(B//64)*4*M*64+row*64,[B//64,4,64],[4*M*64,M*64,1],[B//64,4,64],RL)
 xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(Routed+row*4096,[1,4096],[4096,1],[1,B],XL)
 sd=gl.amd.gfx1250.tdm.make_tensor_descriptor(Shared+row*4096,[1,4096],[4096,1],[1,B],XL)
 gl.amd.gfx1250.tdm.async_load(rd,[0,0,0],rb)
 gl.amd.gfx1250.tdm.async_load(xd,[0,tile*B],xb)
 gl.amd.gfx1250.tdm.async_load(sd,[0,tile*B],sb)
 gl.amd.gfx1250.tdm.async_wait(0)
 r=gl.convert_layout(rb.load(RLAY).permute((1,0,2)).reshape((4,B)),L).to(gl.float32)
 x=gl.sum(xb.load(L).to(gl.float32),0);sh=gl.sum(sb.load(L).to(gl.float32),0)
 x=(x+sh).to(gl.bfloat16).to(gl.float32)
 h=gl.arange(0,4,layout=gl.SliceLayout(1,L));k=tile*B+gl.arange(0,B,layout=gl.SliceLayout(0,L))
 for j in gl.static_range(4):
  r0=gl.sum(gl.where(h[:,None]==0,r,0),0)
  y=gl.fma(x,gl.load(Post+row*4+j),r0*gl.load(Comb+row*16+j))
  for i in gl.static_range(1,4):
   ri=gl.sum(gl.where(h[:,None]==i,r,0),0);y=gl.fma(ri,gl.load(Comb+row*16+i*4+j),y)
  y=gl.inline_asm_elementwise('v_mov_b32 $0, $1',constraints='=v,v',args=[y],dtype=gl.float32,is_pure=True,pack=1)
  gl.store(Out+row*16384+j*4096+k,y.to(gl.bfloat16))
