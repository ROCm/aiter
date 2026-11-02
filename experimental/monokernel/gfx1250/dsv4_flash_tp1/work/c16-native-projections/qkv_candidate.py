import os
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from native_tiles import gemm,weight_first
from qa_deferred import qa_candidate as split_qa
DEFER_QB=gl.constexpr(int(os.getenv("DEFER_QB","0")))
Q_SPLIT=gl.constexpr(int(os.getenv("Q_SPLIT","16")))
PREVIOUS_QA=gl.constexpr(int(os.getenv("PREVIOUS_QA","0")))
from qa_candidate import finish
from native_norm import qa_norm
NATIVE_NORM=gl.constexpr(int(os.getenv("NATIVE_NORM","1")))
from edge_sync import publish,consume,prefetch_bytes
from device_scope_helpers import dev_load,dev_store
from combined_scaled import sync
ADV=gl.constexpr(int(os.getenv('Q_ADV',os.getenv('ADV','0'))))
LOCAL_QUANT=gl.constexpr(int(os.getenv("LOCAL_QUANT","0")))
QA_BUFFERS=gl.constexpr(int(os.getenv('QA_BUFFERS','4')))
QB_HEAD=gl.constexpr(int(os.getenv("QB_HEAD","0")))
QB_BK=gl.constexpr(int(os.getenv("QB_BK","128")))
QB_BUFFERS=gl.constexpr(int(os.getenv('QB_BUFFERS','3' if int(os.getenv('ADV','0'))&16 else '2')))

@g.jit
def input_quant(X,XQ,XS,M:gl.constexpr):
 p=gl.program_id(0)
 if p<32:
  SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[16,128],[1,0])
  sm=gl.allocate_shared_memory(X.dtype.element_ty,[16,128],SL)
  d=gl.amd.gfx1250.tdm.make_tensor_descriptor(X,[M,4096],[4096,1],[16,128],SL)
  gl.amd.gfx1250.tdm.async_load(d,[0,p*128],sm,cache_modifier=".cg");gl.amd.gfx1250.tdm.async_wait(0)
  L:gl.constexpr=gl.BlockedLayout([1,32],[8,4],[4,1],[1,0])
  v=sm.load(L).to(gl.float32)
  a=gl.maximum(gl.max(gl.abs(v),1),1.e-10)/448.
  e=gl.minimum(((a.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
  sc=(e<<23).to(gl.float32,bitcast=True)
  rr=gl.arange(0,16,layout=gl.SliceLayout(1,L));kk=gl.arange(0,128,layout=gl.SliceLayout(0,L))
  gl.store(XQ+rr[:,None]*4096+p*128+kk[None,:],(v/sc[:,None]).to(gl.float8e4nv),rr[:,None]<M,cache_modifier=".cg")
  gl.store(XS+rr*32+p,e.to(gl.uint8),rr<M,cache_modifier=".cg")

@g.jit(do_not_specialize=["CacheRows"])
def qkv_batch(X,XQ,XS,W,WS,P,NW,Raw,QR,QS,Cos,Sin,KV,Counter,Health,
              QBW,QBS,QBRaw,QOut,M:gl.constexpr,EARLY:gl.constexpr=True,Cache=None,Dest=None,BatchId=None,CacheRows=0,Debug=None,DIRECT_KV:gl.constexpr=False,Positions=None,Audit=None,PREFETCH:gl.constexpr=True,SYNC_STYLE:gl.constexpr=3,PREFETCH_STYLE:gl.constexpr=2,QB_BYTES=33554432,Ready=None,SPLITS:gl.constexpr=16,BN:gl.constexpr=128,LOCAL:gl.constexpr=True,PRE:gl.constexpr=True,SEM:gl.constexpr=False,ATOMIC:gl.constexpr=False,Edges=None,OVERLAP:gl.constexpr=False,Epoch=0,QBTiles=None,BranchTiles=None,LayerTiles=None):

 pid=gl.program_id(0)
 if ADV&12:
  epoch=dev_load(Counter,(768+pid)*64)+1
  gl.barrier()
  dev_store(epoch,Counter,(768+pid)*64)
 else:epoch=0
 if ADV&1:
  if DEFER_QB and QB_HEAD and PREVIOUS_QA:
   WLAY:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
   earlyqb=gl.allocate_shared_memory(QBW.dtype.element_ty,[32,QB_BK*16],WLAY)
   qbdesc=gl.amd.gfx1250.tdm.make_tensor_descriptor(QBW+gl.minimum(pid,63)*512*1024,[32,1024*16],[1024*16,1],[32,QB_BK*16],WLAY)
  else:earlyqb=weight_first(QBW,32768,1024,512 if QB_HEAD else 128,QB_BK if QB_HEAD else 512,gl.minimum(pid,63) if QB_HEAD else pid,pid<64 if QB_HEAD else True)
  if not PREVIOUS_QA:earlyqa=weight_first(W,1536,4096,32,1024,gl.minimum(pid,47),pid<48)
  else:earlyqa=None
 else:earlyqb=None;earlyqa=None
 if PREVIOUS_QA:
  split_qa(X,XQ,XS,W,WS,P,NW,Raw,QR,QS,Cos,Sin,KV,Counter,Health,M,QBW if ADV&2 else None,SPLITS=Q_SPLIT,BN=128,LOCAL=True,PRE=True,QB_TILE=earlyqb if DEFER_QB and QB_HEAD and ADV&1 else None,QB_DESC=qbdesc if DEFER_QB and QB_HEAD and ADV&1 else None)
 else:
  if not LOCAL_QUANT:input_quant(X,XQ,XS,M)
  if LOCAL_QUANT:pass
  elif ADV&8:
   publish(Counter+(100+pid//8)*64,pid<32)
  elif ADV&4:
   publish(Counter+64,pid<32)
   consume(Counter+64,Health,32,pid<48,epoch)
  else:sync(Counter,Health,256,3)
  if pid<48:
   gemm(X if LOCAL_QUANT else XQ,XS,W,WS,Raw,M,1536,4096,16,32,1024,QA_BUFFERS,pid,Early=earlyqa,Ready=Counter+100*64 if ADV&8 and not LOCAL_QUANT else None,Health=Health,Epoch=epoch,ReadyCount=8,LOCAL_QUANT=LOCAL_QUANT)
  if ADV&2 and PREFETCH:prefetch_bytes(QBW,33554432)
  if ADV&12:
   publish(Counter+2*64,pid<32)
   publish(Counter+3*64,(pid>=32)&(pid<48))
   consume(Counter+2*64,Health,32,pid<M,epoch)
   consume(Counter+3*64,Health,16,(pid>=M)&(pid<2*M),epoch)
  else:sync(Counter,Health,256,3)
  if pid<M:
   if NATIVE_NORM:qa_norm(Raw,NW,QR,QS,pid,M)
   else:finish(Raw,NW,Raw,QR,QS,Cos,Sin,KV,M,0,1024,pid,SPLITS=1)
  if pid>=M and pid<2*M:finish(Raw,NW,Raw,QR,QS,Cos,Sin,KV,M,1024,512,pid-M,SPLITS=1)
 if ADV&12:
  publish(Counter+4*64,pid<M)
  consume(Counter+4*64,Health,M,True,epoch)
 else:sync(Counter,Health,256,3)
 if QB_HEAD:
  if pid<64:gemm(QR,QS,QBW,QBS,QOut,M,32768,1024,16,512,QB_BK,QB_BUFFERS,pid,Early=earlyqb,NORM_ROPE=True,Cos=Cos,Sin=Sin)
 else:
  gemm(QR,QS,QBW,QBS,QOut,M,32768,1024,16,128,512,QB_BUFFERS,pid,Early=earlyqb)
  if ADV&8:
   publish(Counter+(5+pid//4)*64,True)
   consume(Counter+(5+pid%64)*64,Health,4,True,epoch)
  else:sync(Counter,Health,256,3)
  L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0]);i=gl.arange(0,512,layout=L)
  for rr in gl.static_range(gl.cdiv(M,4)):
   row=pid//64+rr*4;head=pid%64
   if row<M:
    v=gl.load(QOut+row*32768+head*512+i,cache_modifier=".cg").to(gl.float32)
    y=v*gl.rsqrt(gl.sum(v*v,0)/512+1.e-6)
    other=gl.gather(y,i^1,0)
    cv=gl.load(Cos+row*32+(i-448)//2,i>=448,1.);sv=gl.load(Sin+row*32+(i-448)//2,i>=448,0.)
    gl.store(QOut+row*32768+head*512+i,y*cv+gl.where(i%2==0,-1.,1.)*other*sv,cache_modifier=".cg")
