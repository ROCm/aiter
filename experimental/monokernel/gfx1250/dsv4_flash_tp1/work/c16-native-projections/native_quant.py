"""AITER-style 32 values/thread, four threads per 128-value quant group."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from edge_sync import publish

@g.jit
def quant_groups(X,Cos,Sin,Q,S,M:gl.constexpr,N:gl.constexpr,SPAN:gl.constexpr,ROPE:gl.constexpr=False,Completion=None):
 p=gl.program_id(0);row=p%M;span=p//M
 G:gl.constexpr=SPAN//128
 L:gl.constexpr=gl.BlockedLayout([1,32],[8,4],[4,1],[1,0])
 SL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 sm=gl.allocate_shared_memory(X.dtype.element_ty,[G,128],SL)
 d=gl.amd.gfx1250.tdm.make_tensor_descriptor(X+row*N+span*SPAN,[G,128],[128,1],[G,128],SL)
 gl.amd.gfx1250.tdm.async_load(d,[0,0],sm,cache_modifier=".cg")
 kk=gl.arange(0,128,layout=gl.SliceLayout(0,L));gg=gl.arange(0,G,layout=gl.SliceLayout(1,L))
 if ROPE:
  c=gl.load(Cos+row*32+(kk-64)//2,kk>=64,1.).to(gl.float32)
  s=gl.load(Sin+row*32+(kk-64)//2,kk>=64,0.).to(gl.float32)
 gl.amd.gfx1250.tdm.async_wait(0)
 v=sm.load(L).to(gl.float32)
 if ROPE:
  mate=gl.gather(v,(kk^1)[None,:]+gl.full((G,128),0,gl.int32,L),1)
  rope=(gg%4==3)[:,None]&(kk>=64)[None,:]
  r=v*c[None,:]+gl.where(kk%2==0,1.,-1.)[None,:]*mate*s[None,:]
  v=gl.where(rope,r,v)
 a=gl.maximum(gl.max(gl.abs(v),1),1.e-10)/448.
 e=gl.minimum(((a.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
 sc=(e<<23).to(gl.float32,bitcast=True)
 q=(v/sc[:,None]).to(gl.float8e4nv)
 out=gl.allocate_shared_memory(Q.dtype.element_ty,[G,128],SL);out.store(q)
 od=gl.amd.gfx1250.tdm.make_tensor_descriptor(Q+row*N+span*SPAN,[G,128],[128,1],[G,128],SL)
 gl.amd.gfx1250.tdm.async_store(od,[0,0],out,cache_modifier=".cg")
 gl.store(S+row*(N//128)+span*G+gg,e.to(gl.uint8),cache_modifier=".cg")
 gl.amd.gfx1250.tdm.async_wait(0)
 if Completion is not None:publish(Completion+(p//32)*64,True)

@g.jit
def quant_column(X,Q,S,M:gl.constexpr,N:gl.constexpr,COL):
 L:gl.constexpr=gl.BlockedLayout([1,32],[8,4],[4,1],[1,0])
 SL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 sm=gl.allocate_shared_memory(X.dtype.element_ty,[16,128],SL)
 d=gl.amd.gfx1250.tdm.make_tensor_descriptor(X+COL*128,[M,128],[N,1],[16,128],SL)
 gl.amd.gfx1250.tdm.async_load(d,[0,0],sm,cache_modifier='.cg');gl.amd.gfx1250.tdm.async_wait(0)
 v=sm.load(L).to(gl.float32)
 a=gl.maximum(gl.max(gl.abs(v),1),1.e-10)/448.
 e=gl.minimum(((a.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
 sc=(e<<23).to(gl.float32,bitcast=True)
 out=gl.allocate_shared_memory(Q.dtype.element_ty,[16,128],SL);out.store((v/sc[:,None]).to(gl.float8e4nv))
 od=gl.amd.gfx1250.tdm.make_tensor_descriptor(Q+COL*128,[M,128],[N,1],[16,128],SL)
 gl.amd.gfx1250.tdm.async_store(od,[0,0],out,cache_modifier='.cg')
 rr=gl.arange(0,16,layout=gl.SliceLayout(1,L));gl.store(S+rr*(N//128)+COL,e.to(gl.uint8),rr<M,cache_modifier='.cg')
 gl.amd.gfx1250.tdm.async_wait(0)
