from previous.edge_sync import edge,publish,consume,prefetch_bytes
"""Batched input quant + QA/KV scaled-WMMA + reduction/norm/quant.
All M<=8 rows share each TDM weight tile. Fixed resident256-CTA grid.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync
from masked_prefetch import masked_line_prefetch

@g.jit
def finish(P,NW,Y,Q,S,Cos,Sin,KV,M:gl.constexpr,BASE:gl.constexpr,N:gl.constexpr,ROW,Cache=None,Dest=None,BatchId=None,CacheRows=0,Positions=None,SPLITS:gl.constexpr=16,ATOMIC:gl.constexpr=False):
 L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
 i=gl.arange(0,N,layout=L)
 v=gl.full((N,),0,gl.float32,L)
 for k in gl.static_range(1 if ATOMIC else SPLITS):
  if ATOMIC:v+=gl.amd.gfx1250.buffer_load(P,ROW*1536+BASE+i,cache='.cg')
  else:v+=gl.load(P+k*M*1536+ROW*1536+BASE+i)
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
  kv_value=y*c+gl.where(i%2==0,-1.,1.)*mate*sn
  gl.store(KV+ROW*512+i,kv_value)
  if Cache is not None:
   bid=gl.load(BatchId+ROW)
   if Positions is not None:bid=gl.where(gl.load(Positions+ROW)>=0,bid,-1)
   dest=gl.load(Dest+ROW,bid>=0,-1).to(gl.int64)
   gl.store(Cache+dest*512+i,kv_value,(dest>=0)&(dest<CacheRows))

@g.jit
def prefetch_qb(QBW, pid):
 SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[128,128],[1,0])
 d=gl.amd.gfx1250.tdm.make_tensor_descriptor(QBW,[32768,1024],[1024,1],[128,128],SL)
 for j in gl.static_range(8):gl.amd.gfx1250.tdm.prefetch(d,[pid*128,j*128],pred=pid<256,speculative=False)

@g.jit
def qa_candidate(X,XQ,XS,W,WS,P,NW,Y,Q,S,Cos,Sin,KV,Counter,Health,M:gl.constexpr,QBW=None,Cache=None,Dest=None,BatchId=None,CacheRows=0,Positions=None,SYNC_STYLE:gl.constexpr=3,PREFETCH_STYLE:gl.constexpr=1,QB_BYTES=33554432,SPLITS:gl.constexpr=16,BN:gl.constexpr=128,LOCAL:gl.constexpr=True,PRE:gl.constexpr=True,ATOMIC:gl.constexpr=False,Edges=None,Epoch=0,BranchTiles=None):
 pid=gl.program_id(0)
 if not LOCAL:
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
 if ATOMIC:
  if pid<M:
   z=gl.arange(0,2048,layout=gl.BlockedLayout([1],[32],[4],[0]))
   gl.store(P+pid*1536+z,0.,z<1536)
 if not LOCAL or ATOMIC:sync(Counter,Health,256,SYNC_STYLE)
 tile=pid//SPLITS;split=pid%SPLITS
 J:gl.constexpr=32//SPLITS
 if pid<(1536//BN)*SPLITS:
  SW:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[BN,128],[1,0])
  sw=gl.allocate_shared_memory(gl.float8e4nv,[J,BN,128],SW)
  wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(W+tile*BN*4096+split*J*128,[BN,J*128],[4096,1],[BN,128],SW)
  for j in gl.static_range(J):gl.amd.gfx1250.tdm.async_load(wd,[0,j*128],sw.index(j))
  SX:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[16,128],[1,0])
  SF:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[16,128],[1,0])
  if LOCAL:
   sx=gl.allocate_shared_memory(gl.bfloat16,[J,16,128],SX)
   xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(X+split*J*128,[M,J*128],[4096,1],[16,128],SX)
  else:
   sx=gl.allocate_shared_memory(gl.float8e4nv,[J,16,128],SF)
   xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(XQ+split*J*128,[M,J*128],[4096,1],[16,128],SF)
  for j in gl.static_range(J):gl.amd.gfx1250.tdm.async_load(xd,[0,j*128],sx.index(j))
  gl.amd.gfx1250.tdm.async_wait(0)
  ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,128])
  DA:gl.constexpr=gl.DotOperandLayout(0,ML,16);DB:gl.constexpr=gl.DotOperandLayout(1,ML,16)
  AL:gl.constexpr=gl.to_linear_layout(DA,[16,128]);BL:gl.constexpr=gl.to_linear_layout(DB,[128,BN])
  ASL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DA,[16,4],scale_factor=32)
  BSL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DB,[BN,4],scale_factor=32)
  sr=gl.arange(0,16,layout=gl.SliceLayout(1,ASL));sk=gl.arange(0,4,layout=gl.SliceLayout(0,ASL))
  LQ:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
  acc=gl.full((16,BN),0,gl.float32,ML)
  for j in gl.static_range(J):
   if LOCAL:
    v=sx.index(j).load(LQ).to(gl.float32)
    a=gl.maximum(gl.max(gl.abs(v),1),1.e-10)/448.
    e=gl.minimum(((a.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
    sc=(e<<23).to(gl.float32,bitcast=True)
    aa=gl.convert_layout((v/sc[:,None]).to(gl.float8e4nv),DA)
    ee=gl.convert_layout(e,gl.SliceLayout(1,ASL))
    asc=(ee[:,None]+gl.full((16,4),0,gl.uint32,ASL)).to(gl.uint8)
   else:
    aa=gl.convert_layout(sx.index(j).load(AL),DA)
    asc=gl.load(XS+sr[:,None]*32+split*J+j+sk[None,:]*0,sr[:,None]<M,0)
   bb=gl.convert_layout(sw.index(j).permute([1,0]).load(BL),DB)
   bsc=gl.load(WS+(tile*BN//128)*32+split*J+j)+gl.full((BN,4),0,gl.uint8,BSL)
   acc=gl.amd.gfx1250.wmma_scaled(aa,asc,'e4m3',bb,bsc,'e4m3',acc)
  OL:gl.constexpr=gl.to_linear_layout(ML,[16,BN])
  r=gl.arange(0,16,layout=gl.SliceLayout(1,OL));c=gl.arange(0,BN,layout=gl.SliceLayout(0,OL))
  if ATOMIC:gl.atomic_add(P+r[:,None]*1536+tile*BN+c[None,:],gl.convert_layout(acc,OL),r[:,None]<M,sem='relaxed',scope='gpu')
  else:gl.store(P+split*M*1536+r[:,None]*1536+tile*BN+c[None,:],gl.convert_layout(acc,OL),r[:,None]<M)
 if PRE and QBW is not None:prefetch_qb(QBW,pid)
 if BranchTiles is not None:
  publish(BranchTiles+gl.where(pid<(1024//BN)*SPLITS,0,64),pid<(1536//BN)*SPLITS)
  consume(BranchTiles,Health,(1024//BN)*SPLITS,pid<M,Epoch)
  consume(BranchTiles+64,Health,(512//BN)*SPLITS,(pid>=M)&(pid<2*M),Epoch)
 elif Edges is not None:edge(Edges,Health,(1536//BN)*SPLITS,pid<(1536//BN)*SPLITS,pid<2*M,Epoch=Epoch)
 else:sync(Counter,Health,256,SYNC_STYLE)
 if not PRE and QBW is not None:prefetch_qb(QBW,pid)
 if pid<M:finish(P,NW,Y,Q,S,Cos,Sin,KV,M,0,1024,pid,SPLITS=SPLITS,ATOMIC=ATOMIC)
 if pid>=M and pid<2*M:finish(P,NW,Y,Q,S,Cos,Sin,KV,M,1024,512,pid-M,Cache,Dest,BatchId,CacheRows,Positions,SPLITS=SPLITS,ATOMIC=ATOMIC)
