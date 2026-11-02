"""Native elementwise work partitioning; K-major scales for ordinary GEMMs."""
import os
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
ROPE_ORDERED=gl.constexpr(int(os.getenv("ROPE_ORDERED","2")))
ROPE_FMA=gl.constexpr(int(os.getenv("ROPE_FMA","7")))

@g.jit
def quant_column(X,Q,S,M:gl.constexpr,N:gl.constexpr,COL):
 L:gl.constexpr=gl.BlockedLayout([1,32],[8,4],[gl.num_warps(),1],[1,0])
 rr=gl.arange(0,16,layout=gl.SliceLayout(1,L));cc=gl.arange(0,128,layout=gl.SliceLayout(0,L))
 v=gl.load(X+rr[:,None]*N+COL*128+cc[None,:],rr[:,None]<M,0).to(gl.float32)
 a=gl.maximum(gl.max(gl.abs(v),1),1.e-10)/448.
 e=gl.minimum(((a.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
 inv=((254-e)<<23).to(gl.float32,bitcast=True)
 gl.store(Q+rr[:,None]*N+COL*128+cc[None,:],(v*inv[:,None]).to(gl.float8e4nv),rr[:,None]<M)
 gl.store(S+COL*M+rr,e.to(gl.uint8),rr<M)

@g.jit
def qa_norm(Raw,NW,Q,S,Row,M:gl.constexpr):
 L:gl.constexpr=gl.BlockedLayout([1,8],[32,1],[4,1],[0,1])
 r=gl.arange(0,128,layout=gl.SliceLayout(1,L));c=gl.arange(0,8,layout=gl.SliceLayout(0,L))
 v=gl.load(Raw+Row*1536+r[:,None]*8+c[None,:]).to(gl.float32)
 w=gl.load(NW+r[:,None]*8+c[None,:]).to(gl.float32)
 ss=gl.full((128,),0,gl.float32,gl.SliceLayout(1,L))
 for j in gl.static_range(8):
  vv=gl.sum(gl.where(c[None,:]==j,v,0.),1);ss=gl.fma(vv,vv,ss)
 for j in gl.static_range(5):ss=ss+gl.gather(ss,r^(1<<j),0)
 total=gl.sum(gl.where(r%32==0,ss,0.),0)
 y=v*gl.rsqrt(total/1024+1.e-6)*w
 LQ:gl.constexpr=gl.BlockedLayout([1,32],[8,4],[4,1],[1,0])
 t=gl.convert_layout(y.reshape((8,128)),LQ)
 a=gl.maximum(gl.max(gl.abs(t),1),1.e-10)/448.
 e=gl.minimum(((a.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
 inv=((254-e)<<23).to(gl.float32,bitcast=True)
 rr=gl.arange(0,8,layout=gl.SliceLayout(1,LQ));cc=gl.arange(0,128,layout=gl.SliceLayout(0,LQ))
 gl.store(Q+Row*1024+rr[:,None]*128+cc[None,:],(t*inv[:,None]).to(gl.float8e4nv))
 gl.store(S+rr*M+Row,e.to(gl.uint8))

@g.jit
def qk_rope(Raw,QB,NW,Cos,Sin,QOut,KV,Cache,Dest,Bid,M:gl.constexpr,CACHE_ROWS:gl.constexpr,Task):
 L:gl.constexpr=gl.BlockedLayout([1,16],[1,32],[4,1],[1,0])
 rr=gl.arange(0,4,layout=gl.SliceLayout(1,L))+Task//65*4
 cc=gl.arange(0,512,layout=gl.SliceLayout(0,L));head=Task%65
 ptr=QB+rr[:,None]*32768+head*512+cc[None,:]
 if head==64:ptr=Raw+rr[:,None]*1536+1024+cc[None,:]
 v=gl.load(ptr,rr[:,None]<M,0).to(gl.float32)
 # Match native vector reduce followed by descending XOR wave reduction.
 L3:gl.constexpr=gl.BlockedLayout([1,1,16],[1,32,1],[4,1,1],[2,1,0])
 sq=gl.convert_layout(v*v,L).reshape((4,32,16))
 sq=gl.convert_layout(sq,L3)
 if ROPE_ORDERED==2:
  vals=sq
  for step in gl.static_range(4):
   lo,hi=gl.split(vals.reshape((4,32,2,8>>step)).permute((0,1,3,2)))
   vals=lo+hi
  ss=gl.sum(vals,2)
 elif ROPE_ORDERED:
  ss=gl.full((4,32),0.,gl.float32,gl.SliceLayout(2,L3))
  for j in gl.static_range(16):ss=ss+gl.sum(gl.gather(sq,gl.full((4,32,1),j,gl.int32,L3),2),2)
 else:ss=gl.sum(sq,2)
 ss=gl.convert_layout(ss,gl.SliceLayout(2,L3))
 lanes=gl.arange(0,32,layout=gl.SliceLayout(0,gl.SliceLayout(2,L3)))
 for i in gl.static_range(5):ss=ss+gl.gather(ss,(lanes^(16>>i))[None,:]+gl.full((4,32),0,gl.int32,gl.SliceLayout(2,L3)),1)
 total=gl.convert_layout((ss[:,:,None]+gl.full((4,32,16),0,gl.float32,L3)).reshape((4,512)),L)
 if head==64:
  w=gl.load(NW+cc).to(gl.float32);v=v*w[None,:]
 v=v*gl.rsqrt(gl.fma(total,1./512,1.e-6) if ROPE_FMA & 1 else total/512+1.e-6)
 mate=gl.gather(v,(cc^1)[None,:]+gl.full((4,512),0,gl.int32,L),1)
 CL:gl.constexpr=gl.BlockedLayout([1,8],[1,32],[4,1],[1,0])
 pp=gl.arange(0,256,layout=gl.SliceLayout(0,CL))
 cr=gl.arange(0,4,layout=gl.SliceLayout(1,CL))+Task//65*4
 cv=gl.load(Cos+cr[:,None]*32+(pp[None,:]-224),(cr[:,None]<M)&(pp[None,:]>=224),1.).to(gl.float32)
 sv=gl.load(Sin+cr[:,None]*32+(pp[None,:]-224),(cr[:,None]<M)&(pp[None,:]>=224),0.).to(gl.float32)
 c=gl.convert_layout(gl.join(cv,cv).reshape((4,512)),L)
 s=gl.convert_layout(gl.join(sv,sv).reshape((4,512)),L)
 if ROPE_FMA & 4:z=gl.fma(v,c,gl.where(cc%2==0,-1.,1.)[None,:]*mate*s)
 elif ROPE_FMA & 2:z=gl.where((cc%2==0)[None,:],gl.fma(v,c,-mate*s),gl.fma(mate,s,v*c))
 else:z=v*c+gl.where(cc%2==0,-1.,1.)[None,:]*mate*s
 z=z.to(gl.bfloat16)
 if head==64:
  gl.store(KV+rr[:,None]*512+cc[None,:],z,rr[:,None]<M)
  dest=gl.load(Dest+rr,rr<M,-1);bid=gl.load(Bid+rr,rr<M,-1)
  gl.store(Cache+dest[:,None]*512+cc[None,:],z,((rr<M)&(bid>=0)&(dest>=0)&(dest<CACHE_ROWS))[:,None])
 else:gl.store(QOut+rr[:,None]*32768+head*512+cc[None,:],z,rr[:,None]<M)

@g.jit
def inverse_quant(X,Cos,Sin,Q,S,M:gl.constexpr,Task):
 row=Task%M;span=(Task//M)%4;group=Task//(M*4)
 L:gl.constexpr=gl.BlockedLayout([1,32],[8,4],[1,1],[1,0])
 gg=gl.arange(0,8,layout=gl.SliceLayout(1,L));kk=gl.arange(0,128,layout=gl.SliceLayout(0,L))
 SL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 sm=gl.allocate_shared_memory(X.dtype.element_ty,[8,128],SL)
 base=row*32768+group*4096+span*1024
 d=gl.amd.gfx1250.tdm.make_tensor_descriptor(X+base,[8,128],[128,1],[8,128],SL)
 gl.amd.gfx1250.tdm.async_load(d,[0,0],sm)
 CL:gl.constexpr=gl.SliceLayout(0,gl.BlockedLayout([1,16],[8,4],[1,1],[1,0]))
 pp=gl.arange(0,64,layout=CL)
 cv=gl.load(Cos+row*32+pp-32,pp>=32,1.).to(gl.float32)
 sv=gl.load(Sin+row*32+pp-32,pp>=32,0.).to(gl.float32)
 c=gl.convert_layout(gl.join(cv,cv).reshape((128,)),gl.SliceLayout(0,L))
 s=gl.convert_layout(gl.join(sv,sv).reshape((128,)),gl.SliceLayout(0,L))
 gl.amd.gfx1250.tdm.async_wait(0)
 v=sm.load(L).to(gl.float32)
 mate=gl.gather(v,(kk^1)[None,:]+gl.full((8,128),0,gl.int32,L),1)
 z=v*c[None,:]+gl.where(kk%2==0,1.,-1.)[None,:]*mate*s[None,:]
 v=gl.where((gg%4==3)[:,None]&(kk>=64)[None,:],z,v)
 a=gl.maximum(gl.max(gl.abs(v),1),1.e-10)/448.
 e=gl.minimum(((a.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
 inv=((254-e)<<23).to(gl.float32,bitcast=True)
 gl.store(Q+base+gg[:,None]*128+kk[None,:],(v*inv[:,None]).to(gl.float8e4nv))
 gl.store(S+row*256+group*32+span*8+gg,e.to(gl.uint8))

@g.jit
def inverse_quant_group(X,Cos,Sin,Q,S,M:gl.constexpr,Task):
 row=Task%M;span=0;group=Task//M
 L:gl.constexpr=gl.BlockedLayout([1,32],[8,4],[4,1],[1,0])
 gg=gl.arange(0,32,layout=gl.SliceLayout(1,L));kk=gl.arange(0,128,layout=gl.SliceLayout(0,L))
 SL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 sm=gl.allocate_shared_memory(X.dtype.element_ty,[32,128],SL)
 base=row*32768+group*4096+0
 d=gl.amd.gfx1250.tdm.make_tensor_descriptor(X+base,[32,128],[128,1],[32,128],SL)
 gl.amd.gfx1250.tdm.async_load(d,[0,0],sm)
 CL:gl.constexpr=gl.SliceLayout(0,gl.BlockedLayout([1,16],[8,4],[4,1],[1,0]))
 pp=gl.arange(0,64,layout=CL)
 cv=gl.load(Cos+row*32+pp-32,pp>=32,1.).to(gl.float32)
 sv=gl.load(Sin+row*32+pp-32,pp>=32,0.).to(gl.float32)
 c=gl.convert_layout(gl.join(cv,cv).reshape((128,)),gl.SliceLayout(0,L))
 s=gl.convert_layout(gl.join(sv,sv).reshape((128,)),gl.SliceLayout(0,L))
 gl.amd.gfx1250.tdm.async_wait(0)
 v=sm.load(L).to(gl.float32)
 mate=gl.gather(v,(kk^1)[None,:]+gl.full((32,128),0,gl.int32,L),1)
 z=v*c[None,:]+gl.where(kk%2==0,1.,-1.)[None,:]*mate*s[None,:]
 v=gl.where((gg%4==3)[:,None]&(kk>=64)[None,:],z,v)
 a=gl.maximum(gl.max(gl.abs(v),1),1.e-10)/448.
 e=gl.minimum(((a.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
 inv=((254-e)<<23).to(gl.float32,bitcast=True)
 gl.store(Q+base+gg[:,None]*128+kk[None,:],(v*inv[:,None]).to(gl.float8e4nv))
 gl.store(S+row*256+group*32+0+gg,e.to(gl.uint8))

@g.jit
def quant_entry(X,Q,S,M:gl.constexpr,N:gl.constexpr):quant_column(X,Q,S,M,N,gl.program_id(0))
@g.jit
def norm_entry(Raw,NW,Q,S,M:gl.constexpr):qa_norm(Raw,NW,Q,S,gl.program_id(0),M)
@g.jit
def rope_entry(Raw,QB,NW,Cos,Sin,QOut,KV,Cache,Dest,Bid,M:gl.constexpr,CACHE_ROWS:gl.constexpr):qk_rope(Raw,QB,NW,Cos,Sin,QOut,KV,Cache,Dest,Bid,M,CACHE_ROWS,gl.program_id(0))
@g.jit
def inverse_entry(X,Cos,Sin,Q,S,M:gl.constexpr):inverse_quant(X,Cos,Sin,Q,S,M,gl.program_id(0))
