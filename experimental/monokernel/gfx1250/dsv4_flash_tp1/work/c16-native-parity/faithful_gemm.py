"""Native gfx1250 MX128 geometry and pipeline; independent of earlier tuning flags."""
import os
from triton import constexpr_function
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
TDM_CACHE=gl.constexpr(os.getenv("TDM_CACHE",""))

@constexpr_function
def packed_layout(layout, bn):
 def cv(bases):return [[n//16,(k//16)*256+(n%16)*16+k%16] for k,n in bases]
 return gl.DistributedLinearLayout(cv(layout.reg_bases),cv(layout.lane_bases),cv(layout.warp_bases),cv(layout.block_bases),[bn//16,2048])

@g.jit
def wslot(W,E,I):
 if E is None:ret=W.index(I)
 else:
  if I==0:ret=E
  else:ret=W.index(I-1)
 return ret

@g.jit
def early_weight(W,N:gl.constexpr,K:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,NT,Pred=True):
 SL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 sm=gl.allocate_shared_memory(W.dtype.element_ty,[BN//16,BK*16],SL)
 d=gl.amd.gfx1250.tdm.make_tensor_descriptor(W+NT*BN*K,[BN//16,K*16],[K*16,1],[BN//16,BK*16],SL)
 if Pred:gl.amd.gfx1250.tdm.async_load(d,[0,0],sm,warp_used_hint=2,cache_modifier=TDM_CACHE)
 return sm

@g.jit
def operands(A,B,SA,SB,BM:gl.constexpr,BN:gl.constexpr,J:gl.constexpr,ML:gl.constexpr,GROUPED:gl.constexpr):
 DA:gl.constexpr=gl.DotOperandLayout(0,ML,16);DB:gl.constexpr=gl.DotOperandLayout(1,ML,16)
 AL:gl.constexpr=gl.to_linear_layout(DA,[BM,128]);BL:gl.constexpr=gl.to_linear_layout(DB,[128,BN])
 av=gl.amd.cdna4.async_copy.load_shared_relaxed(A.slice(J*128,128,1),AL)
 PL:gl.constexpr=packed_layout(BL,BN)
 bv=gl.amd.cdna4.async_copy.load_shared_relaxed(B.slice(J*2048,2048,1),PL)
 bv=bv.reshape((BN//16,8,16,16)).permute((0,2,1,3)).reshape((BN,128)).trans()
 ASL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DA,[BM,4],scale_factor=32)
 BSL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DB,[BN,4],scale_factor=32)
 AL1:gl.constexpr=gl.to_linear_layout(ASL,[BM,1])
 if GROUPED:aa=gl.sum(SA.slice(J,1,1).load(AL1).to(gl.uint32),1)
 else:
  LL:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,gl.num_warps()],[1,0])
  aa=gl.sum(SA.slice(J,1,0).load(LL).to(gl.uint32),0)
 LL2:gl.constexpr=gl.BlockedLayout([1,1],[32,1],[gl.num_warps(),1],[0,1])
 bs=gl.sum(SB.slice(J,1,1).load(LL2).to(gl.uint32),1)
 asc=gl.convert_layout(aa,gl.SliceLayout(1,ASL))[:,None]+gl.full((BM,4),0,gl.uint32,ASL)
 bsc=gl.sum(bs,0)+gl.full((BN,4),0,gl.uint32,BSL)
 return gl.convert_layout(av,DA),gl.convert_layout(bv,DB),asc.to(gl.uint8),bsc.to(gl.uint8)

@g.jit
def issue(xd,wd,sad,sbd,ab,wb,sa,sb,slot,kt,BK:gl.constexpr,GROUPED:gl.constexpr,Early=None):
 gl.amd.gfx1250.tdm.async_load(xd,[0,kt*BK],ab.index(slot),warp_used_hint=1,cache_modifier=TDM_CACHE)
 if Early is None or kt!=0:gl.amd.gfx1250.tdm.async_load(wd,[0,kt*BK*16],wslot(wb,Early,slot),warp_used_hint=2,cache_modifier=TDM_CACHE)
 if GROUPED:gl.amd.gfx1250.tdm.async_load(sad,[0,kt*(BK//128)],sa.index(slot),warp_used_hint=1<<(2%gl.num_warps()),cache_modifier=TDM_CACHE)
 else:gl.amd.gfx1250.tdm.async_load(sad,[kt*(BK//128),0],sa.index(slot),warp_used_hint=1<<(2%gl.num_warps()),cache_modifier=TDM_CACHE)
 gl.amd.gfx1250.tdm.async_load(sbd,[0,kt*(BK//128)],sb.index(slot),warp_used_hint=1<<(3%gl.num_warps()),cache_modifier=TDM_CACHE)

@g.jit
def compute(ab,wb,sa,sb,slot,acc,xd,wd,sad,sbd,kt,NB:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,ML:gl.constexpr,GROUPED:gl.constexpr,REFILL:gl.constexpr,Early=None):
 for j in gl.static_range(BK//128):
  aa,bb,asc,bsc=operands(ab.index(slot),wslot(wb,Early,slot),sa.index(slot),sb.index(slot),BM,BN,j,ML,GROUPED)
  acc=gl.amd.gfx1250.wmma_scaled(aa,asc,'e4m3',bb,bsc,'e4m3',acc)
  if REFILL and j==0:issue(xd,wd,sad,sbd,ab,wb,sa,sb,(kt+NB-1)%NB,kt+NB-1,BK,GROUPED,Early)
 return acc

@g.jit
def gemm(X,XS,W,WS,Y,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,NB:gl.constexpr,NT,GROUPED:gl.constexpr=False,Early=None):
 ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1]] if gl.num_warps()==2 else ([[0,1],[1,0]] if BM==32 else [[0,1],[0,2]]),instr_shape=[16,16,128])
 WA:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 XA:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[BK,16]],[BM,BK],[1,0])
 SC:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 wb=gl.allocate_shared_memory(W.dtype.element_ty,[NB if Early is None else NB-1,BN//16,BK*16],WA)
 ab=gl.allocate_shared_memory(X.dtype.element_ty,[NB,BM,BK],XA)
 sa=gl.allocate_shared_memory(gl.uint8,[NB,BM,BK//128] if GROUPED else [NB,BK//128,BM],SC)
 sb=gl.allocate_shared_memory(gl.uint8,[NB,1,BK//128],SC)
 xb=(NT//(1024//BN))*K if GROUPED else 0
 xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(X+xb,[M,K],[32768 if GROUPED else K,1],[BM,BK],XA)
 wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(W+NT*BN*K,[BN//16,K*16],[K*16,1],[BN//16,BK*16],WA)
 if GROUPED:sad=gl.amd.gfx1250.tdm.make_tensor_descriptor(XS+xb//128,[M,K//128],[256,1],[BM,BK//128],SC)
 else:sad=gl.amd.gfx1250.tdm.make_tensor_descriptor(XS,[K//128,M],[M,1],[BK//128,BM],SC)
 sbd=gl.amd.gfx1250.tdm.make_tensor_descriptor(WS+(NT*BN//128)*(K//128),[1,K//128],[K//128,1],[1,BK//128],SC)
 for j in gl.static_range(NB-1):issue(xd,wd,sad,sbd,ab,wb,sa,sb,j,j,BK,GROUPED,Early)
 acc=gl.full((BM,BN),0,gl.float32,ML)
 for kt in range(K//BK-(NB-1)):
  gl.amd.gfx1250.tdm.async_wait(NB-2);gl.barrier()
  acc=compute(ab,wb,sa,sb,kt%NB,acc,xd,wd,sad,sbd,kt,NB,BM,BN,BK,ML,GROUPED,True,Early)
 for j in gl.static_range(NB-1):
  kt=K//BK-(NB-1)+j
  gl.amd.gfx1250.tdm.async_wait(NB-2-j);gl.barrier()
  acc=compute(ab,wb,sa,sb,kt%NB,acc,xd,wd,sad,sbd,kt,NB,BM,BN,BK,ML,GROUPED,False,Early)
 gl.amd.gfx1250.tdm.async_wait(0);gl.barrier()
 YL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[BN,8]],[BM,BN],[1,0]) if BN>=128 else WA
 out=gl.allocate_shared_memory(Y.dtype.element_ty,[BM,BN],YL)
 out.store(acc.to(gl.bfloat16))
 yd=gl.amd.gfx1250.tdm.make_tensor_descriptor(Y+NT*BN,[M,N],[N,1],[BM,BN],YL)
 gl.amd.gfx1250.tdm.async_store(yd,[0,0],out,cache_modifier=TDM_CACHE)
 gl.amd.gfx1250.tdm.async_wait(0)

@g.jit
def entry(X,XS,W,WS,Y,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,NB:gl.constexpr,GROUPED:gl.constexpr=False):
 gemm(X,XS,W,WS,Y,M,N,K,BM,BN,BK,NB,gl.program_id(0),GROUPED)
