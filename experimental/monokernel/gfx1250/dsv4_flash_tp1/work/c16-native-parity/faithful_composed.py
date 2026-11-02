import os
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
QA_WS=gl.constexpr(int(os.getenv("QA_WS","1")))
OB_NB=gl.constexpr(int(os.getenv("FAITHFUL_OB_BUFFERS","4")))
ADV=gl.constexpr(int(os.getenv("FAITHFUL_ADV","0")))
from faithful_gemm import gemm,early_weight
from edge_sync import prefetch_bytes
from faithful_ops import quant_column,qa_norm,qk_rope,inverse_quant,inverse_quant_group
from edge_sync import publish,consume
from device_scope_helpers import dev_load,dev_store

@g.jit
def idle():pass

@g.jit
def qa_prefix(X,XQ,XS,W,WS,Raw,C,H,Epoch,M:gl.constexpr,STAGE:gl.constexpr):
 p=gl.program_id(0)
 if p<32:quant_column(X,XQ,XS,M,4096,p)
 if STAGE>=2:
  publish(C,p<32);consume(C,H,32,p<48,Epoch)
  if p<48:gemm(XQ,XS,W,WS,Raw,M,1536,4096,16,32,1024,4,p)

@g.jit
def qkv(X,XQ,XS,W,WS,Raw,NW,Q,QS,QBW,QBS,QBR,KNW,Cos,Sin,QO,KV,Cache,Dest,Bid,C,H,Epochs,M:gl.constexpr,STAGE:gl.constexpr):
 p=gl.program_id(0);epoch=dev_load(Epochs,p)+1;dev_store(epoch,Epochs,p)
 if ADV & 1 and STAGE>=4:prefetch_bytes(QBW,32768*1024)
 if ADV & 2 and STAGE>=4:ew=early_weight(QBW,32768,1024,128,512,p)
 else:ew=None
 if STAGE<=2:qa_prefix(X,XQ,XS,W,WS,Raw,C,H,epoch,M,STAGE)
 else:
  if QA_WS:gl.warp_specialize([(idle,()),(qa_prefix,(X,XQ,XS,W,WS,Raw,C,H,epoch,M,STAGE))],[2])
  else:qa_prefix(X,XQ,XS,W,WS,Raw,C,H,epoch,M,STAGE)
  publish(C+64,p<32);consume(C+64,H,32,p<M,epoch)
  if p<M:qa_norm(Raw,NW,Q,QS,p,M)
  if STAGE>=4:
   publish(C+128,p<M);consume(C+128,H,M,True,epoch)
   gemm(Q,QS,QBW,QBS,QBR,M,32768,1024,16,128,512,2,p,Early=ew)
  if STAGE>=5:
   # Each head consumes exactly four QB tiles, for all sixteen rows.
   publish(C+256+(p//4)*64,True)
   # KV has sixteen QA producers; distinct from QA's thirty-two producers.
   publish(C+192,(p>=32)&(p<48))
   for t in range(p,260,256):
    if t%65==64:consume(C+192,H,16,True,epoch)
    else:consume(C+256+(t%65)*64,H,4,True,epoch)
    qk_rope(Raw,QBR,KNW,Cos,Sin,QO,KV,Cache,Dest,Bid,M,M*128,t)

@g.jit
def inv_worker(X,Cos,Sin,Q,S,M:gl.constexpr):
 p=gl.program_id(0)
 for t in range(p,512,256):inverse_quant(X,Cos,Sin,Q,S,M,t)

@g.jit
def quant_worker(X,Q,S,M:gl.constexpr):
 p=gl.program_id(0)
 if p<64:quant_column(X,Q,S,M,8192,p)

@g.jit
def output(X,Cos,Sin,XQ,XS,WA,SA,A,AQ,AS,WB,SB,Y,C,H,Epochs,M:gl.constexpr,STAGE:gl.constexpr):
 p=gl.program_id(0);epoch=dev_load(Epochs,p)+1;dev_store(epoch,Epochs,p)
 gl.warp_specialize([(idle,()),(inv_worker,(X,Cos,Sin,XQ,XS,M))],[1])
 if STAGE>=2:
  publish(C,True);consume(C,H,256,True,epoch)
  gemm(XQ,XS,WA,SA,A,M,8192,4096,32,32,1024,4,p,True)
 if STAGE>=3:
  publish(C+64,True);consume(C+64,H,256,p<64,epoch)
  gl.warp_specialize([(idle,()),(quant_worker,(A,AQ,AS,M))],[2])
 if STAGE>=4:
  publish(C+128,p<64);consume(C+128,H,64,p<64,epoch)
  if p<64:gemm(AQ,AS,WB,SB,Y,M,4096,8192,16,64,512,OB_NB,p)

@g.jit
def compact_output(X,Cos,Sin,XQ,XS,WA,SA,A,AQ,AS,WB,SB,Y,C,H,Epochs,M:gl.constexpr,STAGE:gl.constexpr):
 p=gl.program_id(0);epoch=dev_load(Epochs,p)+1;dev_store(epoch,Epochs,p)
 if ADV & 1:
  prefetch_bytes(WA,8192*4096)
  prefetch_bytes(WB,4096*8192)
 if ADV & 2:
  ewa=early_weight(WA,8192,4096,32,1024,p)
  ewb=early_weight(WB,4096,8192,64,512,p,p<64)
 else:ewa=None;ewb=None
 if p<128:inverse_quant_group(X,Cos,Sin,XQ,XS,M,p)
 if STAGE>=2:
  publish(C+(p//16)*64,p<128)
  consume(C+(p//32)*64,H,16,True,epoch)
  gemm(XQ,XS,WA,SA,A,M,8192,4096,32,32,1024,4,p,True,Early=ewa)
 if STAGE>=3:
  publish(C+512+(p//4)*64,True)
  consume(C+512+p*64,H,4,p<64,epoch)
  if p<64:quant_column(A,AQ,AS,M,8192,p)
 if STAGE>=4:
  publish(C+4608,p<64);consume(C+4608,H,64,p<64,epoch)
  if p<64:gemm(AQ,AS,WB,SB,Y,M,4096,8192,16,64,512,OB_NB,p,Early=ewb)
