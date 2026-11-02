"""Experimental C1 full layer, including compressed attention state transitions.
Supported specialization: BF16 main cache, FP8 index cache, context <= 2048.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync,complete
from layout import unshuffle_body
from tdm_tail import candidate_body as mhc_body
from output_scaled import output_body
from post import post_body
from attention_extended import cache_attention_body
from ffn_extended import ffn_body
from project_compress import project_body
from compressor import compressor_body
from identity_indexer import identity_csa

@g.jit
def model_layer(A,F,MC,IC,MW,IW,MY,IY,Positions,CosTable,SinTable,
                RealCache,RealIndices,RealPtr,Dest,Topk,Blocks,Bias,
                MM:gl.constexpr,IM:gl.constexpr,RATIO:gl.constexpr,ENVELOPE:gl.constexpr,
                W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,
                HAS_BIAS:gl.constexpr,T:gl.constexpr=640):
 p=gl.program_id(0)
 unshuffle_body(A[0],A[1])
 sync(A[29],A[30],256,3)
 # Preserve QA L2 prefetch. QB TDM and prefetch remain inside complete().
 if p<192:
  tid=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
  addr=(A[12]+(p//16)*128*4096+(p%16)*256+tid*4096).to(gl.uint64)
  gl.inline_asm_elementwise('global_prefetch_b8 $1, off scope:SCOPE_DEV th:TH_LOAD_NT\n v_mov_b32 $0,0',constraints='=v,v,~{memory}',args=[addr],dtype=gl.int32,is_pure=False,pack=1)
 mhc_body(A[1],A[3],A[4],A[5],A[6],A[7],A[8],A[9],A[10],A[29],A[30])
 sync(A[29],A[30],256,3)
 if RATIO!=0:
  N:gl.constexpr=2048 if RATIO==4 else 1024
  project_body(A[10],MW,MY,N)
  if RATIO==4:
   project_body(A[10],IW,IY,512)
  sync(A[29],A[30],256,3)
  compressor_body(MY,MY+N//2,*MC,*MM,OWNER=0)
  if RATIO==4:
   compressor_body(IY,IY+256,*IC,*IM,OWNER=1)
   identity_csa(Positions,RealPtr,Blocks,RealIndices,Topk,A[30],ENVELOPE)
  sync(A[29],A[30],256,3)
 n=gl.load(RealPtr+1)-gl.load(RealPtr)
 if p==0:
  i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]));pos=gl.load(Positions)
  gl.store(A[21]+i,gl.load(CosTable+pos*32+i));gl.store(A[22]+i,gl.load(SinTable+pos*32+i))
  gl.store(A[33],0);gl.store(A[33]+1,n)
 # Stage a compact attention cache, preserving the native compressed-first order.
 for part in gl.static_range(gl.cdiv(T,256)):
  row=p+part*256
  if row<T:
   SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,512],[1,0])
   sm=gl.allocate_shared_memory(gl.bfloat16,[1,512],SL)
   slot=gl.load(RealIndices+gl.load(RealPtr)+row,row<n,0)
   desc=gl.amd.gfx1250.tdm.make_tensor_descriptor(RealCache+gl.maximum(slot,0).to(gl.int64)*512,[1,512],[512,1],[1,512],SL)
   gl.amd.gfx1250.tdm.async_load(desc,[0,0],sm);gl.amd.gfx1250.tdm.async_wait(0)
   L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0]);i=gl.arange(0,512,layout=L)
   v=gl.convert_layout(gl.sum(sm.load(gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])).to(gl.float32),0),L)
   gl.store(A[31]+row*512+i,v)
   gl.store(A[32]+row,gl.where(row==n-1,T-1,row))
 sync(A[29],A[30],256,3)
 complete(A[10],A[11],A[12],A[13],A[14],A[15],A[16],A[17],A[18],A[19],A[20],A[21],A[22],A[23],A[24],A[25],A[26],A[27],A[28],A[29],A[30],BN=128,GRID=256,STYLE=3,U=4,QAMODE=0,ALLTDM=True,EARLY=True,PREFETCH=2)
 sync(A[29],A[30],256,3)
 cache_attention_body(A[28],A[23],A[31],A[32],A[33],A[34],A[35],A[36],A[37],A[38],A[39],A[29],A[30],T,T,T-1,4,256,16,32,False,True)
 sync(A[29],A[30],256,3)
 output_body(A[38],A[21],A[22],*A[40:],A[29],A[30])
 sync(A[29],A[30],256,3)
 post_body(A[1],A[51],A[7],A[2],128,False,2)
 sync(F[-2],F[-1],256,3)
 ffn_body(*F,W1S,S1S,W2S,S2S,DO_PRE=True,POST_SHUFFLED=True,Bias=Bias,HAS_BIAS=HAS_BIAS)
 if p==0:
  i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]));dest=gl.load(Dest).to(gl.int64)
  gl.store(RealCache+dest*512+i,gl.load(A[23]+i),dest>=0)
