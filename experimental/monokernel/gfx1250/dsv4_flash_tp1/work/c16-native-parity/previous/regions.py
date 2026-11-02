from previous.edge_sync import edge,publish,consume,prefetch_bytes
from combined_scaled import poll_read
"""True multirow full layer: a fixed resident grid and batched WMMA stages."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync
from previous.layout_batch import unshuffle,shuffle
from previous.mhc_batch import mhc_batch
from previous.mhc_pre_aiter import pre as mhc_pre_aiter
from previous.output_batch import output_body
from previous.post_batch import post_batch
from previous.attention_paged import attention_paged
from previous.ffn_batch import ffn_body
from previous.project_batch import project_body
from previous.optimized_compressor import compressor_body,pool_partial
from previous.mhc_integrated import post_pre
from previous.identity_batch import identity_csa
from previous.qkv_candidate import qkv_batch
from masked_prefetch import masked_line_prefetch

@g.jit
def region(A,F,MC,IC,MW,IW,MY,IY,Positions,CosTable,SinTable,
 RealCache,RealIndices,RealPtr,Dest,Topk,Blocks,Bias,
 XQ,XS,Chosen,Weights,MainPlan,IndexPlan,Output,
 MM:gl.constexpr,IM:gl.constexpr,RATIO:gl.constexpr,ENVELOPE:gl.constexpr,
 W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,
 HAS_BIAS:gl.constexpr,M:gl.constexpr,BLOCK_STRIDE:gl.constexpr,T:gl.constexpr=640,
 FP32IN=None,FP32MID=None,FP32OUT=None,STOP:gl.constexpr=0,PREFETCH:gl.constexpr=True,BatchIds=None,CacheRows=0,Work=None,CompCounter=None,MHC=(),ML:gl.constexpr=(),MV:gl.constexpr=(),OPT_MHC:gl.constexpr=True,OPT_COMP:gl.constexpr=True,DBG=None,PRE_W=None,OPT_PRE:gl.constexpr=False,Edges=None,OVERLAP:gl.constexpr=False,EDGE_MASK:gl.constexpr=31,Tiles=None,TILE_MASK:gl.constexpr=15,GROUP:gl.constexpr=0):
 gl.static_assert(M<=16)
 gl.static_assert(M<=8 or (TILE_MASK&63)==0, "C16 requires non-overlapping mailbox allocation before fine-grained tile modes")
 p=gl.program_id(0)
 Epoch=1
 if GROUP==0:
  p=gl.program_id(0)
  if Tiles is not None and TILE_MASK&63:
   for reset_tile in range(p,1696,256):
    reset_needed=((TILE_MASK&1)!=0)&(reset_tile<32)
    reset_needed=reset_needed|(((TILE_MASK&2)!=0)&(reset_tile>=32)&(reset_tile<200))
    reset_needed=reset_needed|(((TILE_MASK&4)!=0)&(reset_tile>=256)&(reset_tile<1088))
    reset_needed=reset_needed|(((TILE_MASK&8)!=0)&(reset_tile>=200)&(reset_tile<248))
    reset_needed=reset_needed|(((TILE_MASK&16)!=0)&(reset_tile>=248)&(reset_tile<250))
    reset_needed=reset_needed|(((TILE_MASK&32)!=0)&(reset_tile>=1088))
    gl.store(Tiles+reset_tile*64,0,reset_needed)
  if Edges is not None:
   if p<20:gl.store(Edges+p*320,0)
   if p<64:gl.store(Edges+24*320+p*64,0)
  if Edges is not None:
   if p<M:
    i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]));pos=gl.load(Positions+p)
    gl.store(A[21]+p*32+i,gl.load(CosTable+pos*32+i));gl.store(A[22]+p*32+i,gl.load(SinTable+pos*32+i))
  if (OVERLAP & 1) and OPT_PRE:prefetch_bytes(PRE_W,24*16384*4)
  if Tiles is None or not TILE_MASK&64 or not OPT_PRE:unshuffle(A[0],A[1],M)
  sync(A[29],A[30],256,3)
  Epoch=1
  if PREFETCH and p<192:
   PFSL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[128,128],[1,0])
   pfd=gl.amd.gfx1250.tdm.make_tensor_descriptor(A[12],[1536,4096],[4096,1],[128,128],PFSL)
   for pf in gl.static_range(2):
    gl.amd.gfx1250.tdm.prefetch(pfd,[(p//16)*128,(p%16)*256+pf*128],pred=p<192,speculative=False)
  if OPT_PRE:
   mhc_pre_aiter(A[1],A[7],A[10],A[4],A[5],A[6],(PRE_W,)+MHC[1:],A[30],M,ML,MV,FP32IN,Edges if EDGE_MASK&1 else None,Epoch=Epoch,Shuffled=A[0] if Tiles is not None and TILE_MASK&64 else None)
  else:
   mhc_batch(A[1],A[3],A[4],A[5],A[6],A[7],A[8],A[9],A[10],A[29],A[30],M,FP32=FP32IN)
  if Edges is not None and OPT_PRE and EDGE_MASK&1:
   publish(Edges+320,p<M)
   if (OVERLAP & 1) and RATIO!=0:
    prefetch_bytes(MW,(2048 if RATIO==4 else 1024)*4096*2)
    if RATIO==4:prefetch_bytes(IW,512*4096*2)
   consume(Edges+320,A[30],M,p<192,Epoch=Epoch)
  else:sync(A[29],A[30],256,3)
 if GROUP==1:
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
   if Edges is not None and EDGE_MASK&4:
    edge(Edges+2*320,A[30],N//16,p<N//16,p<(2*M if RATIO==4 else 4*M),Epoch=Epoch)
   elif OPT_COMP and RATIO!=128:
    if p<N//16:sync(CompCounter,A[30],N//16,3)
   else:sync(A[29],A[30],256,3)
   if DBG is not None and p==0:gl.store(DBG,gl.load(A[30]))
   if RATIO==128 and OPT_COMP:
    pool_partial(MC[8],MC[9],MainPlan,MC[6],Work,MM[4],MM[5],MM[6],MM[7],M,8)
    if Tiles is not None and TILE_MASK&32:
     publish(Tiles+(1672+p//4)*64,p<M*4)
     consume(Tiles+(1672+p)*64,A[30],4,p<M,1)
    elif Edges is not None and EDGE_MASK&4:edge(Edges+3*320,A[30],M*4,p<M*4,p<M,Epoch=Epoch)
    else:sync(A[29],A[30],256,3)
   if DBG is not None and p==0:gl.store(DBG+1,gl.load(A[30]))
   if p<M:
    CR=MC[:4]+(MainPlan+p*8,MainPlan+p*8+4,MC[6]+p,MC[7]+p)+MC[8:12]+(MC[12]+p*BLOCK_STRIDE,)
    compressor_body(MY+p*N,MY+p*N+N//2,*CR,*MM,OWNER=p,Partial=Work+p*6144 if RATIO==128 and OPT_COMP else None,EARLY=(OVERLAP&8)!=0)
   if RATIO==4:
    if p>=M and p<2*M:
     row=p-M
     IR=IC[:4]+(IndexPlan+row*8,IndexPlan+row*8+4,IC[6]+row,IC[7]+row)+IC[8:12]+(IC[12]+row*BLOCK_STRIDE,)
     compressor_body(IY+row*512,IY+row*512+256,*IR,*IM,OWNER=p,EARLY=(OVERLAP&8)!=0)
    if p<M*4:
     row=p//4
     valid=gl.load(BatchIds+row)>=0 if BatchIds is not None else True
     if valid:identity_csa(Positions+row,RealPtr+row,Blocks+row*BLOCK_STRIDE,RealIndices,Topk+row*512,A[30],ENVELOPE,p%4)
   if Tiles is not None and TILE_MASK&32:publish(Tiles+(1664+(p//4 if RATIO==4 else p))*64,p<(M*4 if RATIO==4 else M))
   elif Edges is not None and EDGE_MASK&4:publish(Edges+4*320,p<(M*4 if RATIO==4 else M))
   else:sync(A[29],A[30],256,3)
  if DBG is not None and p==0:gl.store(DBG+2,gl.load(A[30]))
 if GROUP==2:
  if p<M:
   i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]));pos=gl.load(Positions+p)
   gl.store(A[21]+p*32+i,gl.load(CosTable+pos*32+i));gl.store(A[22]+p*32+i,gl.load(SinTable+pos*32+i))
  sync(A[29],A[30],256,3)
  qkv_batch(A[10],XQ,XS,A[12],A[13],A[14],A[15],A[18],A[19],A[20],A[21],A[22],A[23],A[29],A[30],A[24],A[25],A[26],A[28],M,PREFETCH=PREFETCH,PREFETCH_STYLE=1,Edges=Edges+5*320 if Edges is not None and EDGE_MASK&2 else None,OVERLAP=OVERLAP,Epoch=Epoch,QBTiles=Edges+24*320 if Edges is not None and EDGE_MASK&16 else None,BranchTiles=Tiles+248*64 if Tiles is not None and TILE_MASK&16 else None,LayerTiles=Tiles if TILE_MASK&32 else None)
  if Tiles is None or not TILE_MASK&32:sync(A[29],A[30],256,3)
  if DBG is not None and p==0:gl.store(DBG+3,gl.load(A[30]))
 if GROUP==3:
  if Edges is not None and RATIO!=0 and EDGE_MASK&4 and (Tiles is None or not TILE_MASK&32):consume(Edges+4*320,A[30],M*4 if RATIO==4 else M,True,Epoch=Epoch)
  if OVERLAP & 1:
   prefetch_bytes(A[40],8192*4096)
   prefetch_bytes(A[41],8192*4096//128//128)
  attention_paged(A[28],A[23],RealCache,RealIndices,RealPtr,A[34],A[35],A[36],A[37],A[38],A[39],A[29],A[30],Dest,CacheRows,M,(4 if M==16 else 8),Tiles if TILE_MASK&1 else None,LayerTiles=Tiles if TILE_MASK&32 else None,COMP=4 if RATIO==4 else (1 if RATIO else 0))
  if Tiles is None or not TILE_MASK&32:sync(A[29],A[30],256,3)
 if GROUP==4:
  if OVERLAP & 1:
   prefetch_bytes(MHC[0],24*16384*4)
   prefetch_bytes(F[5],4096*2)
  output_body(A[38],A[21],A[22],*A[40:],A[29],A[30],M,STOP=STOP-10 if STOP>10 else 0,OVERLAP=OVERLAP,Tiles=Tiles if TILE_MASK&2 else None,InputTiles=Tiles+1136*64 if Tiles is not None and TILE_MASK&32 else None)
  if Tiles is None or not TILE_MASK&2 or not OPT_MHC:sync(A[29],A[30],256,3)
 if GROUP==5:
  if OPT_MHC:
   post_pre(A[51],A[1],A[7],F[0],F[6],F[9],F[3],F[4],F[5],MHC,F[-1],M,ML,MV,Edges+7*320 if Edges is not None and EDGE_MASK&1 else None,Epoch=Epoch,InputTiles=Tiles+168*64 if Tiles is not None and TILE_MASK&2 else None)
   if Edges is not None and EDGE_MASK&1:edge(Edges+8*320,F[-1],M,p<M,True,Epoch=Epoch)
   else:sync(F[-2],F[-1],256,3)
  else:
   post_batch(A[1],A[51],A[7],A[2],M,False,FP32MID)
   sync(F[-2],F[-1],256,3)
 if GROUP==6:
  RunF=F[:1]+(Output,)+F[2:] if Tiles is not None and TILE_MASK&128 else F
  ffn_body(*RunF,W1S,S1S,W2S,S2S,M,Chosen,Weights,DO_PRE=not OPT_MHC,POST_SHUFFLED=Tiles is not None and (TILE_MASK&128)!=0,Bias=Bias,HAS_BIAS=HAS_BIAS,FP32IN=FP32MID,FP32OUT=FP32OUT,Edges=Edges if EDGE_MASK&8 else None,OVERLAP=OVERLAP,Epoch=Epoch,Tiles=Tiles,TILE_MASK=TILE_MASK)
  if Tiles is None or not TILE_MASK&128:
   sync(F[-2],F[-1],256,3)
   shuffle(F[1],Output,M)
  if p<M:
   i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]));dest=gl.load(Dest+p).to(gl.int64)
   valid=gl.load(BatchIds+p)>=0 if BatchIds is not None else True
   gl.store(RealCache+dest*512+i,gl.load(A[23]+p*512+i),(dest>=0)&(dest<CacheRows)&valid)
