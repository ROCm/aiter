"""Correctness control: one resident grid processes decode rows sequentially.
Not the final batched-WMMA optimization. Never launch M*256 spinning CTAs.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from model_layer_precise import model_layer as row_layer
from combined_scaled import sync

@g.jit
def batched_layer(A,F,MC,IC,MW,IW,MY,IY,Positions,CosTable,SinTable,
                  RealCache,RealIndices,RealPtr,Dest,Topk,Blocks,Bias,
                  MainPlan,IndexPlan,RowInput,RowOutput,
                  MM:gl.constexpr,IM:gl.constexpr,RATIO:gl.constexpr,ENVELOPE:gl.constexpr,
                  W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,
                  HAS_BIAS:gl.constexpr,M:gl.constexpr,BLOCK_STRIDE:gl.constexpr,
                  T:gl.constexpr=640,FP32IN=None,FP32MID=None,FP32OUT=None):
 for row in range(M):
  if gl.program_id(0)==0:
   L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
   i=gl.arange(0,8,layout=L);pos=gl.load(Positions+row)
   if RATIO:
    cp=gl.where((pos+1)%RATIO==0,pos,-1)
    v=gl.where(i==2,cp,gl.where(i==3,RATIO*(2 if RATIO==4 else 1)-1,gl.where(i==6,pos,0)))
    gl.store(MainPlan+i,v)
   if RATIO==4:
    cp=gl.where((pos+1)%4==0,pos,-1)
    v=gl.where(i==2,cp,gl.where(i==3,7,gl.where(i==6,pos,0)))
    gl.store(IndexPlan+i,v)
  sync(A[29],A[30],256,3)
  # ATOM residual storage is [K/64, HC, M, 64], not row-contiguous.
  pid=gl.program_id(0)
  if pid<64:
   SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[4,64],[1,0])
   LL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
   sm=gl.allocate_shared_memory(gl.bfloat16,[4,64],SL)
   desc=gl.amd.gfx1250.tdm.make_tensor_descriptor(A[0]+pid*256*M+row*64,[4,64],[M*64,1],[4,64],SL)
   gl.amd.gfx1250.tdm.async_load(desc,[0,0],sm);gl.amd.gfx1250.tdm.async_wait(0)
   h=gl.arange(0,4,layout=gl.SliceLayout(1,LL));k=gl.arange(0,64,layout=gl.SliceLayout(0,LL))
   gl.store(RowInput+pid*256+h[:,None]*64+k[None,:],sm.load(LL))
  sync(A[29],A[30],256,3)
  AR=(RowInput,)+A[1:]
  FR=(F[0],RowOutput)+F[2:]
  if RATIO:
   CR=MC[:4]+(MainPlan,MainPlan+4,MC[6]+row,MC[7]+row)+MC[8:12]+(MC[12]+row*BLOCK_STRIDE,)
  else:CR=MC
  if RATIO==4:
   IR=IC[:4]+(IndexPlan,IndexPlan+4,IC[6]+row,IC[7]+row)+IC[8:12]+(IC[12]+row*BLOCK_STRIDE,)
  else:IR=IC
  if FP32IN is not None:pin=FP32IN+row*16384
  else:pin=None
  row_layer(AR,FR,CR,IR,MW,IW,MY,IY,Positions+row,CosTable,SinTable,
            RealCache,RealIndices,RealPtr+row,Dest+row,Topk,Blocks+row*BLOCK_STRIDE,Bias,
            MM,IM,RATIO,ENVELOPE,W1S,S1S,W2S,S2S,HAS_BIAS,T,
            FP32IN=pin,FP32MID=FP32MID+row*16384,FP32OUT=FP32OUT+row*16384)
  sync(A[29],A[30],256,3)
  if gl.program_id(0)<64:
   pid=gl.program_id(0)
   SL2:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[4,64],[1,0])
   LL2:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
   sm2=gl.allocate_shared_memory(gl.bfloat16,[4,64],SL2)
   desc2=gl.amd.gfx1250.tdm.make_tensor_descriptor(RowOutput+pid*256,[4,64],[64,1],[4,64],SL2)
   gl.amd.gfx1250.tdm.async_load(desc2,[0,0],sm2);gl.amd.gfx1250.tdm.async_wait(0)
   h2=gl.arange(0,4,layout=gl.SliceLayout(1,LL2));k2=gl.arange(0,64,layout=gl.SliceLayout(0,LL2))
   gl.store(F[1]+pid*256*M+h2[:,None]*M*64+row*64+k2[None,:],sm2.load(LL2))
  sync(A[29],A[30],256,3)
