from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

@g.jit
def qa_norm(Raw,NW,Q,S,Row,M:gl.constexpr):
 SL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 sm=gl.allocate_shared_memory(Raw.dtype.element_ty,[128,8],SL)
 d=gl.amd.gfx1250.tdm.make_tensor_descriptor(Raw+Row*1536,[128,8],[8,1],[128,8],SL)
 gl.amd.gfx1250.tdm.async_load(d,[0,0],sm,cache_modifier='.cg')
 L:gl.constexpr=gl.BlockedLayout([1,8],[32,1],[4,1],[0,1])
 r=gl.arange(0,128,layout=gl.SliceLayout(1,L));c=gl.arange(0,8,layout=gl.SliceLayout(0,L))
 w=gl.load(NW+r[:,None]*8+c[None,:]).to(gl.float32)
 gl.amd.gfx1250.tdm.async_wait(0)
 v=sm.load(L).to(gl.float32)
 ss=gl.full((128,),0,gl.float32,gl.SliceLayout(1,L))
 for j in gl.static_range(8):
  vv=gl.sum(gl.where(c[None,:]==j,v,0.),1)
  ss=gl.fma(vv,vv,ss)
 for j in gl.static_range(7):ss=ss+gl.gather(ss,r^(1<<j),0)
 total=gl.sum(gl.where(r==0,ss,0.),0)
 y=v*gl.rsqrt(total/1024+1.e-6)*w
 LQ:gl.constexpr=gl.BlockedLayout([1,32],[8,4],[4,1],[1,0])
 t=gl.convert_layout(y.reshape((8,128)),LQ)
 a=gl.maximum(gl.max(gl.abs(t),1),1.e-10)/448.
 e=gl.minimum(((a.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
 sc=(e<<23).to(gl.float32,bitcast=True)
 qs=gl.allocate_shared_memory(Q.dtype.element_ty,[8,128],SL);qs.store((t/sc[:,None]).to(gl.float8e4nv))
 qd=gl.amd.gfx1250.tdm.make_tensor_descriptor(Q+Row*1024,[8,128],[128,1],[8,128],SL)
 gl.amd.gfx1250.tdm.async_store(qd,[0,0],qs,cache_modifier=".cg")
 rr=gl.arange(0,8,layout=gl.SliceLayout(1,LQ));gl.store(S+Row*8+rr,e.to(gl.uint8),cache_modifier=".cg")
 gl.amd.gfx1250.tdm.async_wait(0)
