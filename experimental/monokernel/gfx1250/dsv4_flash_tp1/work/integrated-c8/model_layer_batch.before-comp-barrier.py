"""True multirow full layer: a fixed resident grid and batched WMMA stages."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync
from layout_batch import unshuffle,shuffle
from mhc_batch import mhc_batch
from output_batch import output_body
from post_batch import post_batch
from attention_paged import attention_paged
from ffn_batch import ffn_body
from project_batch import project_body
from optimized_compressor import compressor_body,pool_partial
from mhc_integrated import post_pre
from identity_batch import identity_csa
from qkv_candidate import qkv_batch
from masked_prefetch import masked_line_prefetch

@g.jit
def model_layer(A,F,MC,IC,MW,IW,MY,IY,Positions,CosTable,SinTable,
 RealCache,RealIndices,RealPtr,Dest,Topk,Blocks,Bias,
 XQ,XS,Chosen,Weights,MainPlan,IndexPlan,Output,
 MM:gl.constexpr,IM:gl.constexpr,RATIO:gl.constexpr,ENVELOPE:gl.constexpr,
 W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,
 HAS_BIAS:gl.constexpr,M:gl.constexpr,BLOCK_STRIDE:gl.constexpr,T:gl.constexpr=640,
 FP32IN=None,FP32MID=None,FP32OUT=None,STOP:gl.constexpr=0,PREFETCH:gl.constexpr=True,BatchIds=None,CacheRows=0,Work=None,CompCounter=None,MHC=(),ML:gl.constexpr=(),MV:gl.constexpr=(),OPT_MHC:gl.constexpr=True,OPT_COMP:gl.constexpr=True,DBG=None):
 p=gl.program_id(0)
 unshuffle(A[0],A[1],M)
 sync(A[29],A[30],256,3)
 if PREFETCH and p<192:
  PFSL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[128,128],[1,0])
  pfd=gl.amd.gfx1250.tdm.make_tensor_descriptor(A[12],[1536,4096],[4096,1],[128,128],PFSL)
  for pf in gl.static_range(2):
   gl.amd.gfx1250.tdm.prefetch(pfd,[(p//16)*128,(p%16)*256+pf*128],pred=p<192,speculative=False)
 mhc_batch(A[1],A[3],A[4],A[5],A[6],A[7],A[8],A[9],A[10],A[29],A[30],M,FP32=FP32IN)
 sync(A[29],A[30],256,3)
 if STOP==1:return
 if RATIO!=0:
  N:gl.constexpr=2048 if RATIO==4 else 1024
  project_body(A[10],MW,MY,N,M)
  if RATIO==4:project_body(A[10],IW,IY,512,M)
  if p<M:
   i=gl.arange(0,8,layout=gl.BlockedLayout([1],[32],[4],[0]));pos=gl.load(Positions+p)
   valid=gl.load(BatchIds+p)>=0 if BatchIds is not None else True
   cp=gl.where(valid & ((pos+1)%RATIO==0),pos,-1)
   v=gl.where(i==2,cp,gl.where(i==3,RATIO*(2 if RATIO==4 else 1)-1,gl.where(i==6,gl.where(valid,pos,-1),0)))
   gl.store(MainPlan+p*8+i,v)
   if RATIO==4:gl.store(IndexPlan+p*8+i,v)
  if OPT_COMP:
   if p<N//16:sync(CompCounter,A[30],N//16,3)
  else:sync(A[29],A[30],256,3)
  if DBG is not None and p==0:gl.store(DBG,gl.load(A[30]))
  if RATIO==128 and OPT_COMP:
   pool_partial(MC[8],MC[9],MainPlan,MC[6],Work,MM[4],MM[5],MM[6],MM[7],M,8)
   if p<N//16:sync(CompCounter,A[30],N//16,3)
  if DBG is not None and p==0:gl.store(DBG+1,gl.load(A[30]))
  if p<M:
   CR=MC[:4]+(MainPlan+p*8,MainPlan+p*8+4,MC[6]+p,MC[7]+p)+MC[8:12]+(MC[12]+p*BLOCK_STRIDE,)
   compressor_body(MY+p*N,MY+p*N+N//2,*CR,*MM,OWNER=p,Partial=Work+p*6144 if RATIO==128 and OPT_COMP else None)
  if RATIO==4:
   if p>=M and p<2*M:
    row=p-M
    IR=IC[:4]+(IndexPlan+row*8,IndexPlan+row*8+4,IC[6]+row,IC[7]+row)+IC[8:12]+(IC[12]+row*BLOCK_STRIDE,)
    compressor_body(IY+row*512,IY+row*512+256,*IR,*IM,OWNER=p)
   if p<M*4:
    row=p//4
    valid=gl.load(BatchIds+row)>=0 if BatchIds is not None else True
    if valid:identity_csa(Positions+row,RealPtr+row,Blocks+row*BLOCK_STRIDE,RealIndices,Topk+row*512,A[30],ENVELOPE,p%4)
  sync(A[29],A[30],256,3)
 if DBG is not None and p==0:gl.store(DBG+2,gl.load(A[30]))
 if p<M:
  i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]));pos=gl.load(Positions+p)
  gl.store(A[21]+p*32+i,gl.load(CosTable+pos*32+i));gl.store(A[22]+p*32+i,gl.load(SinTable+pos*32+i))
 sync(A[29],A[30],256,3)
 qkv_batch(A[10],XQ,XS,A[12],A[13],A[14],A[15],A[18],A[19],A[20],A[21],A[22],A[23],A[29],A[30],A[24],A[25],A[26],A[28],M,PREFETCH=PREFETCH,PREFETCH_STYLE=1)
 sync(A[29],A[30],256,3)
 if DBG is not None and p==0:gl.store(DBG+3,gl.load(A[30]))
 if STOP==2:return
 attention_paged(A[28],A[23],RealCache,RealIndices,RealPtr,A[34],A[35],A[36],A[37],A[38],A[39],A[29],A[30],Dest,CacheRows,M,8)
 sync(A[29],A[30],256,3)
 if STOP==3:return
 output_body(A[38],A[21],A[22],*A[40:],A[29],A[30],M,STOP=STOP-10 if STOP>10 else 0)
 sync(A[29],A[30],256,3)
 if STOP==4 or STOP>10:return
 if OPT_MHC:
  post_pre(A[51],A[1],A[7],F[0],F[6],F[9],F[3],F[4],F[5],MHC,F[-1],M,ML,MV)
  sync(F[-2],F[-1],256,3)
 else:
  post_batch(A[1],A[51],A[7],A[2],M,False,FP32MID)
  sync(F[-2],F[-1],256,3)
 if STOP==5:return
 ffn_body(*F,W1S,S1S,W2S,S2S,M,Chosen,Weights,DO_PRE=not OPT_MHC,POST_SHUFFLED=False,Bias=Bias,HAS_BIAS=HAS_BIAS,FP32IN=FP32MID,FP32OUT=FP32OUT)
 sync(F[-2],F[-1],256,3)
 shuffle(F[1],Output,M)
 if p<M:
  i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]));dest=gl.load(Dest+p).to(gl.int64)
  valid=gl.load(BatchIds+p)>=0 if BatchIds is not None else True
  gl.store(RealCache+dest*512+i,gl.load(A[23]+p*512+i),(dest>=0)&(dest<CacheRows)&valid)
