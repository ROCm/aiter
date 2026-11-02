"""Batched input quant + QA/KV scaled-WMMA + reduction/norm/quant.
All M<=8 rows share each TDM weight tile. Fixed resident256-CTA grid.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync

@g.jit
def finish(P,NW,Y,Q,S,Cos,Sin,KV,M:gl.constexpr,BASE:gl.constexpr,N:gl.constexpr,ROW):
 L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
 i=gl.arange(0,N,layout=L)
 v=gl.full((N,),0,gl.float32,L)
 for k in gl.static_range(16):v+=gl.load(P+k*M*1536+ROW*1536+BASE+i)
 v=v.to(gl.bfloat16).to(gl.float32)
 if BASE==0:
  # Match AITER HIP:8 contiguous values/thread, FMA accumulation, then
  # ascending pair reductions within each wave and across its4 waves.
  tid=gl.arange(0,128,layout=L)
  ss=gl.full((128,),0,gl.float32,L)
  for j in gl.static_range(8):
   vv=gl.gather(v,tid*8+j,0)
   ss=gl.fma(vv,vv,ss)
  for j in gl.static_range(7):ss=ss+gl.gather(ss,tid^(1<<j),0)
  total=gl.sum(gl.where(tid==0,ss,0.),0)
 else:total=gl.sum(v*v,0)
 y=v*gl.rsqrt(total/N+1.e-6)*gl.load(NW+BASE+i)
 gl.store(Y+ROW*1536+BASE+i,v)
 if BASE==0:
  LQ:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[0,1])
  t=gl.convert_layout(gl.reshape(y,(8,128)),LQ)
  a=gl.maximum(gl.max(gl.abs(t),1),1.e-10)/448.
  e=gl.minimum(((a.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
  scale=(e<<23).to(gl.float32,bitcast=True)
  r=gl.arange(0,8,layout=gl.SliceLayout(1,LQ));c=gl.arange(0,128,layout=gl.SliceLayout(0,LQ))
  gl.store(Q+ROW*1024+r[:,None]*128+c[None,:],(t/scale[:,None]).to(gl.float8e4nv));gl.store(S+ROW*8+r,e.to(gl.uint8))
 else:
  mate=gl.gather(y,i^1,0)
  c=gl.load(Cos+ROW*32+(i-448)//2,i>=448,1.)
  sn=gl.load(Sin+ROW*32+(i-448)//2,i>=448,0.)
  gl.store(KV+ROW*512+i,y*c+gl.where(i%2==0,-1.,1.)*mate*sn)

@g.jit
def qa_batch(X,XQ,XS,W,WS,P,NW,Y,Q,S,Cos,Sin,KV,Counter,Health,M:gl.constexpr,QBW=None):
 pid=gl.program_id(0)
 # Input tiles use TDM; each CTA quantizes the same K group for all rows.
 if pid<32:
  SM:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[16,128],[1,0])
  sm=gl.allocate_shared_memory(gl.bfloat16,[16,128],SM)
  desc=gl.amd.gfx1250.tdm.make_tensor_descriptor(X,[M,4096],[4096,1],[16,128],SM)
  gl.amd.gfx1250.tdm.async_load(desc,[0,pid*128],sm);gl.amd.gfx1250.tdm.async_wait(0)
  L:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
  v=sm.load(L).to(gl.float32)
  a=gl.maximum(gl.max(gl.abs(v),1),1.e-10)/448.
  e=gl.minimum(((a.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
  sc=(e<<23).to(gl.float32,bitcast=True)
  r=gl.arange(0,16,layout=gl.SliceLayout(1,L));k=gl.arange(0,128,layout=gl.SliceLayout(0,L))
  gl.store(XQ+r[:,None]*4096+pid*128+k[None,:],(v/sc[:,None]).to(gl.float8e4nv),r[:,None]<M)
  gl.store(XS+r*32+pid,e.to(gl.uint8),r<M)
 # Prefetch both weight tiles while producers finish input quantization.
 tile=pid//16;split=pid%16
 SLW:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[128,128],[1,0])
 sw=gl.allocate_shared_memory(gl.float8e4nv,[2,128,128],SLW)
 if pid<192:
  wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(W+tile*128*4096+split*256,[128,256],[4096,1],[128,128],SLW)
  for j in gl.static_range(2):gl.amd.gfx1250.tdm.async_load(wd,[0,j*128],sw.index(j))
 sync(Counter,Health,256,3)
 if pid<192:
  ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,128])
  DA:gl.constexpr=gl.DotOperandLayout(0,ML,16);DB:gl.constexpr=gl.DotOperandLayout(1,ML,16)
  AL:gl.constexpr=gl.to_linear_layout(DA,[16,128]);BL:gl.constexpr=gl.to_linear_layout(DB,[128,128])
  ASL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DA,[16,4],scale_factor=32)
  BSL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DB,[128,4],scale_factor=32)
  SA:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[16,128],[1,0])
  sx=gl.allocate_shared_memory(gl.float8e4nv,[2,16,128],SA)
  xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(XQ+split*256,[M,256],[4096,1],[16,128],SA)
  for j in gl.static_range(2):gl.amd.gfx1250.tdm.async_load(xd,[0,j*128],sx.index(j))
  gl.amd.gfx1250.tdm.async_wait(0)
  acc=gl.full((16,128),0,gl.float32,ML)
  sr=gl.arange(0,16,layout=gl.SliceLayout(1,ASL));sk=gl.arange(0,4,layout=gl.SliceLayout(0,ASL))
  for j in gl.static_range(2):
   aa=gl.convert_layout(sx.index(j).load(AL),DA)
   bb=gl.convert_layout(sw.index(j).permute([1,0]).load(BL),DB)
   asc=gl.load(XS+sr[:,None]*32+split*2+j+sk[None,:]*0,sr[:,None]<M,0)
   bsc=gl.load(WS+tile*32+split*2+j)+gl.full((128,4),0,gl.uint8,BSL)
   acc=gl.amd.gfx1250.wmma_scaled(aa,asc,'e4m3',bb,bsc,'e4m3',acc)
  OL:gl.constexpr=gl.to_linear_layout(ML,[16,128])
  r=gl.arange(0,16,layout=gl.SliceLayout(1,OL));c=gl.arange(0,128,layout=gl.SliceLayout(0,OL))
  gl.store(P+split*M*1536+r[:,None]*1536+tile*128+c[None,:],gl.convert_layout(acc,OL),r[:,None]<M)
 sync(Counter,Health,256,3)
 if QBW is not None:
  tid=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
  for pf in gl.static_range(4):
   addr=(QBW+((pf*256+pid)*128+tid)*256).to(gl.uint64)
   gl.inline_asm_elementwise('global_prefetch_b8 $1, off scope:SCOPE_DEV th:TH_LOAD_NT\n v_mov_b32 $0,0',constraints='=v,v,~{memory}',args=[addr],dtype=gl.int32,is_pure=False,pack=1)
 if pid<M:finish(P,NW,Y,Q,S,Cos,Sin,KV,M,0,1024,pid)
 if pid>=M and pid<2*M:finish(P,NW,Y,Q,S,Cos,Sin,KV,M,1024,512,pid-M)
