"""True multirow full layer: a fixed resident grid and batched WMMA stages."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync
from layout_batch import unshuffle,shuffle
from mhc_batch import mhc_batch
from output_batch import output_body
from post_batch import post_batch
from attention_batch import attention_body
from ffn_batch import ffn_body
from project_batch import project_body
from compressor_batch import compressor_body
from identity_batch import identity_csa
from qkv_full_batch import qkv_batch
from masked_prefetch import masked_line_prefetch

@g.jit
def model_layer(A,F,MC,IC,MW,IW,MY,IY,Positions,CosTable,SinTable,
 RealCache,RealIndices,RealPtr,Dest,Topk,Blocks,Bias,
 XQ,XS,Chosen,Weights,MainPlan,IndexPlan,Output,
 MM:gl.constexpr,IM:gl.constexpr,RATIO:gl.constexpr,ENVELOPE:gl.constexpr,
 W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,
 HAS_BIAS:gl.constexpr,M:gl.constexpr,BLOCK_STRIDE:gl.constexpr,T:gl.constexpr=640,
 FP32IN=None,FP32MID=None,FP32OUT=None,STOP:gl.constexpr=0,PREFETCH:gl.constexpr=True,BatchIds=None):
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
  sync(A[29],A[30],256,3)
  if p<M:
   CR=MC[:4]+(MainPlan+p*8,MainPlan+p*8+4,MC[6]+p,MC[7]+p)+MC[8:12]+(MC[12]+p*BLOCK_STRIDE,)
   compressor_body(MY+p*N,MY+p*N+N//2,*CR,*MM,OWNER=p)
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
 if p<M:
  i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]));pos=gl.load(Positions+p)
  gl.store(A[21]+p*32+i,gl.load(CosTable+pos*32+i));gl.store(A[22]+p*32+i,gl.load(SinTable+pos*32+i))
  n=gl.load(RealPtr+p+1)-gl.load(RealPtr+p)
  gl.store(A[33]+p*2,0);gl.store(A[33]+p*2+1,n)
 for task in range(p,M*T,256):
  token=task//T;row=task%T
  start=gl.load(RealPtr+token);n=gl.load(RealPtr+token+1)-start
  SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,512],[1,0])
  sm=gl.allocate_shared_memory(gl.bfloat16,[1,512],SL)
  slot=gl.load(RealIndices+start+row,row<n,0)
  desc=gl.amd.gfx1250.tdm.make_tensor_descriptor(RealCache+gl.maximum(slot,0).to(gl.int64)*512,[1,512],[512,1],[1,512],SL)
  gl.amd.gfx1250.tdm.async_load(desc,[0,0],sm);gl.amd.gfx1250.tdm.async_wait(0)
  L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0]);i=gl.arange(0,512,layout=L)
  v=gl.convert_layout(gl.sum(sm.load(gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])).to(gl.float32),0),L)
  gl.store(A[31]+task*512+i,v)
  gl.store(A[32]+task,gl.where(row==n-1,T-1,row))
 sync(A[29],A[30],256,3)
 qkv_batch(A[10],XQ,XS,A[12],A[13],A[14],A[15],A[18],A[19],A[20],A[21],A[22],A[23],A[29],A[30],A[24],A[25],A[26],A[28],M,PREFETCH=PREFETCH,PREFETCH_STYLE=1)
 sync(A[29],A[30],256,3)
 if STOP==2:return
 attention_body(A[28],A[23],A[31],A[32],A[33],A[34],A[35],A[36],A[37],A[38],A[39],A[29],A[30],M,T,64)
 sync(A[29],A[30],256,3)
 if STOP==3:return
 output_body(A[38],A[21],A[22],*A[40:],A[29],A[30],M,STOP=STOP-10 if STOP>10 else 0)
 sync(A[29],A[30],256,3)
 if STOP==4 or STOP>10:return
 post_batch(A[1],A[51],A[7],A[2],M,False,FP32MID)
 sync(F[-2],F[-1],256,3)
 if STOP==5:return
 ffn_body(*F,W1S,S1S,W2S,S2S,M,Chosen,Weights,DO_PRE=True,POST_SHUFFLED=False,Bias=Bias,HAS_BIAS=HAS_BIAS,FP32IN=FP32MID,FP32OUT=FP32OUT)
 sync(F[-2],F[-1],256,3)
 shuffle(F[1],Output,M)
 if p<M:
  i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]));dest=gl.load(Dest+p).to(gl.int64)
  valid=gl.load(BatchIds+p)>=0 if BatchIds is not None else True
  gl.store(RealCache+dest*512+i,gl.load(A[23]+p*512+i),(dest>=0)&valid)
