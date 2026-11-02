"""Gluon adaptation of gfx1250 FlyDSL MX128 tuned GEMMs.
Packed B: [N/16,K/16,16,16]. No split-K; FP32 accumulate, BF16 epilogue.
TDM ring contains activation, weight, and both scale tiles. A/B register
fragments are fetched one WMMA K step ahead of use.
"""
import os
from triton import constexpr_function
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from edge_sync import consume
PIPE_REG = gl.constexpr(int(os.getenv('PIPE_REG','1')))
DIRECT_PACKED = gl.constexpr(int(os.getenv('DIRECT_PACKED','1')))

@constexpr_function
def packed_layout(layout, bn):
 def convert(bases):
  return [[n//16,(k//16)*256+(n%16)*16+k%16] for k,n in bases]
 return gl.DistributedLinearLayout(convert(layout.reg_bases),convert(layout.lane_bases),convert(layout.warp_bases),convert(layout.block_bases),[bn//16,2048])

TDM_SCALE = gl.constexpr(int(os.getenv('TDM_SCALE','1')))

@g.jit
def weight_first(W,N:gl.constexpr,K:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,NT,Pred=True):
 SL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 sm=gl.allocate_shared_memory(W.dtype.element_ty,[BN//16,BK*16],SL)
 d=gl.amd.gfx1250.tdm.make_tensor_descriptor(W+NT*BN*K,[BN//16,K*16],[K*16,1],[BN//16,BK*16],SL)
 if Pred:gl.amd.gfx1250.tdm.async_load(d,[0,0],sm,cache_modifier=".cg")
 return sm

@g.jit
def wslot(WB,E,I):
 if E is not None:
  if I==0:ret=E
  else:ret=WB.index(I-1)
 else:ret=WB.index(I)
 return ret

@g.jit
def tile_operands(A,B,SA,SB,XScale,WScale,BM:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,J:gl.constexpr,ML:gl.constexpr,M:gl.constexpr,XSROW:gl.constexpr,XSCOL,WSROW,K:gl.constexpr,KI):
 DA:gl.constexpr=gl.DotOperandLayout(0,ML,16);DB:gl.constexpr=gl.DotOperandLayout(1,ML,16)
 AL:gl.constexpr=gl.to_linear_layout(DA,[BM,128]);BL:gl.constexpr=gl.to_linear_layout(DB,[128,BN])
 av=gl.amd.cdna4.async_copy.load_shared_relaxed(A.slice(J*128,128,1),AL)
 PL:gl.constexpr=packed_layout(BL,BN) if DIRECT_PACKED else gl.BlockedLayout([1,16],[1,32],[4,1],[1,0])
 packed=gl.amd.cdna4.async_copy.load_shared_relaxed(B.slice(J*2048,2048,1),PL)
 bv=packed.reshape((BN//16,8,16,16)).permute((0,2,1,3)).reshape((BN,128)).trans()
 ASL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DA,[BM,4],scale_factor=32)
 BSL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DB,[BN,4],scale_factor=32)
 if TDM_SCALE:
  LL:gl.constexpr=gl.BlockedLayout([1,1],[32,1],[4,1],[0,1])
  aa=gl.sum(SA.slice(J,1,1).load(LL).to(gl.uint32),1)
  bs=gl.sum(SB.slice(J,1,1).load(LL).to(gl.uint32),1)
  asc=gl.convert_layout(aa,gl.SliceLayout(1,ASL))[:,None]+gl.full((BM,4),0,gl.uint32,ASL)
  if BN>128:
   bl=gl.convert_layout(bs,gl.SliceLayout(1,BSL))
   nr=gl.arange(0,BN,layout=gl.SliceLayout(1,BSL))
   brs=gl.gather(bl,nr//128,0)
   bsc=brs[:,None]+gl.full((BN,4),0,gl.uint32,BSL)
  else:bsc=gl.sum(bs,0)+gl.full((BN,4),0,gl.uint32,BSL)
 else:
  rr=gl.arange(0,BM,layout=gl.SliceLayout(1,ASL));kk=gl.arange(0,4,layout=gl.SliceLayout(0,ASL))
  asc=gl.load(XScale+rr[:,None]*XSROW+XSCOL+KI*(BK//128)+J+kk[None,:]*0,rr[:,None]<M,0)
  bsc=gl.load(WScale+WSROW*(K//128)+KI*(BK//128)+J)+gl.full((BN,4),0,gl.uint8,BSL)
 return gl.convert_layout(av,DA),gl.convert_layout(bv,DB),asc.to(gl.uint8),bsc.to(gl.uint8)

@g.jit
def gemm(X,XS,W,WS,Y,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,BUFFERS:gl.constexpr,NT,GROUPED:gl.constexpr=False,Early=None,Ready=None,Health=None,Epoch=0,ReadyCount:gl.constexpr=16,LOCAL_QUANT:gl.constexpr=False,NORM_ROPE:gl.constexpr=False,Cos=None,Sin=None):
 NTILES:gl.constexpr=K//BK
 NB:gl.constexpr=min(BUFFERS,NTILES)
 DEPTH:gl.constexpr=min(BUFFERS-1,NTILES)
 OPS:gl.constexpr=(3 if LOCAL_QUANT else 4) if TDM_SCALE else 2
 ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[1,0]] if BN==32 else [[0,1],[0,2]],instr_shape=[16,16,128])
 WA:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 XA:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[BK,16]],[BM,BK],[1,0])
 XB:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[BK//2,8]],[BM,BK],[1,0]) if LOCAL_QUANT else XA
 SC:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 # The early tile is ring slot zero; never copy it through registers.
 wb=gl.allocate_shared_memory(W.dtype.element_ty,[NB-1 if Early is not None else NB,BN//16,BK*16],WA)
 ab=gl.allocate_shared_memory(X.dtype.element_ty,[NB,BM,BK],XB)
 sa=gl.allocate_shared_memory(gl.uint8,[NB,BM,BK//128],SC)
 sb=gl.allocate_shared_memory(gl.uint8,[NB,max(1,BN//128),BK//128],SC)
 xb=(NT//(1024//BN))*K if GROUPED else 0
 xstride:gl.constexpr=32768 if GROUPED else K
 xsstride:gl.constexpr=256 if GROUPED else K//128
 wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(W+NT*BN*K,[BN//16,K*16],[K*16,1],[BN//16,BK*16],WA)
 xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(X+xb,[M,K],[xstride,1],[BM,BK],XB)
 sad=gl.amd.gfx1250.tdm.make_tensor_descriptor(XS+xb//128,[M,K//128],[xsstride,1],[BM,BK//128],SC)
 sbd=gl.amd.gfx1250.tdm.make_tensor_descriptor(WS+(NT*BN//128)*(K//128),[max(1,BN//128),K//128],[K//128,1],[max(1,BN//128),BK//128],SC)
 for j in gl.static_range(DEPTH):
  if Ready is not None:consume(Ready+j*64,Health,ReadyCount,True,Epoch)
  if j!=0 or Early is None:gl.amd.gfx1250.tdm.async_load(wd,[0,j*BK*16],wslot(wb,Early,j),cache_modifier=".cg")
  gl.amd.gfx1250.tdm.async_load(xd,[0,j*BK],ab.index(j),cache_modifier=".cg")
  if TDM_SCALE:
   if not LOCAL_QUANT:gl.amd.gfx1250.tdm.async_load(sad,[0,j*(BK//128)],sa.index(j),cache_modifier=".cg")
   gl.amd.gfx1250.tdm.async_load(sbd,[0,j*(BK//128)],sb.index(j),cache_modifier=".cg")
 if LOCAL_QUANT:
  aq=gl.allocate_shared_memory(gl.float8e4nv,[BM,BK],XA)
 acc=gl.full((BM,BN),0,gl.float32,ML)
 for b in range(NTILES):
  if b<NTILES-DEPTH+1:gl.amd.gfx1250.tdm.async_wait((DEPTH-1)*OPS)
  else:gl.amd.gfx1250.tdm.async_wait(0)
  # Refilling a free slot preserves the current tile for all its K steps.
  if LOCAL_QUANT:
   QL:gl.constexpr=gl.BlockedLayout([1,32],[8,4],[4,1],[1,0])
   FL:gl.constexpr=gl.BlockedLayout([1,32],[1,32],[4,1],[1,0])
   xv=gl.convert_layout(ab.index(b%NB).load(FL).reshape((BM*(BK//128),128)),QL).to(gl.float32)
   mx=gl.maximum(gl.max(gl.abs(xv),1),1.e-10)/448.
   ee=gl.minimum(((mx.to(gl.uint32,bitcast=True)&0x7fffffff)+0x7fffff)>>23,254)
   sc=(ee<<23).to(gl.float32,bitcast=True)
   aq.store((xv/sc[:,None]).to(gl.float8e4nv).reshape((BM,BK)))
   sa.index(b%NB).store(ee.to(gl.uint8).reshape((BM,BK//128)))
   current_a=aq
  else:current_a=ab.index(b%NB)
  aa,bb,asc,bsc=tile_operands(current_a,wslot(wb,Early,b%NB),sa.index(b%NB),sb.index(b%NB),XS,WS,BM,BN,BK,0,ML,M,xsstride,xb//128,NT*BN//128,K,b)
  gl.barrier()
  if b+DEPTH<NTILES:
   if Ready is not None:consume(Ready+(b+DEPTH)*64,Health,ReadyCount,True,Epoch)
   gl.amd.gfx1250.tdm.async_load(wd,[0,(b+DEPTH)*BK*16],wslot(wb,Early,(b+DEPTH)%NB),cache_modifier=".cg")
   gl.amd.gfx1250.tdm.async_load(xd,[0,(b+DEPTH)*BK],ab.index((b+DEPTH)%NB),cache_modifier=".cg")
   if TDM_SCALE:
    if not LOCAL_QUANT:gl.amd.gfx1250.tdm.async_load(sad,[0,(b+DEPTH)*(BK//128)],sa.index((b+DEPTH)%NB),cache_modifier=".cg")
    gl.amd.gfx1250.tdm.async_load(sbd,[0,(b+DEPTH)*(BK//128)],sb.index((b+DEPTH)%NB),cache_modifier=".cg")
  for j in gl.static_range(BK//128):
   if PIPE_REG and j+1<BK//128:
    an,bn,asn,bsn=tile_operands(current_a,wslot(wb,Early,b%NB),sa.index(b%NB),sb.index(b%NB),XS,WS,BM,BN,BK,j+1,ML,M,xsstride,xb//128,NT*BN//128,K,b)
   acc=gl.amd.gfx1250.wmma_scaled(aa,asc,'e4m3',bb,bsc,'e4m3',acc)
   if j+1<BK//128:
    if PIPE_REG:aa=an;bb=bn;asc=asn;bsc=bsn
    else:aa,bb,asc,bsc=tile_operands(current_a,wslot(wb,Early,b%NB),sa.index(b%NB),sb.index(b%NB),XS,WS,BM,BN,BK,j+1,ML,M,xsstride,xb//128,NT*BN//128,K,b)
 gl.amd.gfx1250.tdm.async_wait(0)
 gl.barrier()
 YL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[BN,8]],[BM,BN],[1,0])
 out=gl.allocate_shared_memory(Y.dtype.element_ty,[BM,BN],YL)
 if NORM_ROPE:
  NL:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])
  vv=gl.convert_layout(acc.to(gl.bfloat16).to(gl.float32),NL)
  vv=vv*gl.rsqrt(gl.sum(vv*vv,1)/512+1.e-6)[:,None]
  rr=gl.arange(0,BM,layout=gl.SliceLayout(1,NL));kk=gl.arange(0,512,layout=gl.SliceLayout(0,NL))
  mate=gl.gather(vv,(kk^1)[None,:]+gl.full((BM,512),0,gl.int32,NL),1)
  cv=gl.load(Cos+rr[:,None]*32+(kk[None,:]-448)//2,(rr[:,None]<M)&(kk[None,:]>=448),1.)
  sv=gl.load(Sin+rr[:,None]*32+(kk[None,:]-448)//2,(rr[:,None]<M)&(kk[None,:]>=448),0.)
  out.store((vv*cv+gl.where(kk%2==0,-1.,1.)[None,:]*mate*sv).to(Y.dtype.element_ty))
 else:out.store(acc.to(gl.bfloat16).to(Y.dtype.element_ty))
 yd=gl.amd.gfx1250.tdm.make_tensor_descriptor(Y+NT*BN,[M,N],[N,1],[BM,BN],YL)
 gl.amd.gfx1250.tdm.async_store(yd,[0,0],out,cache_modifier=".cg")
 gl.amd.gfx1250.tdm.async_wait(0)
