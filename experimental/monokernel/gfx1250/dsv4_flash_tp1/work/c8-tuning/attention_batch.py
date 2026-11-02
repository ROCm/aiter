from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from aiter_matched import _pa_decode_sparse, _pa_decode_sparse_reduce
from combined import sync
from masked_prefetch import masked_line_prefetch
@g.jit
def compute_attention(Q,KV,Cache,Indices,Indptr,Sink,M,L,A,O,Counts,T:gl.constexpr,PAGES:gl.constexpr,SPLITS:gl.constexpr,BH:gl.constexpr=16,BK:gl.constexpr=32,WRITE_SLOT:gl.constexpr=0,DIRECT:gl.constexpr=False,PID=0):
    pid=PID
    if pid<(64//BH)*SPLITS:
        h=pid//SPLITS;k=pid%SPLITS
        n=gl.load(Indptr+1)-gl.load(Indptr)
        if k*gl.cdiv(n,SPLITS*BK)<gl.cdiv(n,BK):
            _pa_decode_sparse(Q,Cache,Cache,Indices,Indptr,M,L,A,Sink,O,PAGES,
                64*512,512,1,512,1,1,
                SPLITS*64,64,1,SPLITS*64,64,1,
                SPLITS*64*512,64*512,512,1,64*512,512,1,
                64,512,SPLITS,0.04419417382415922,BH,512,BK,True,False,128,4,True,4,h,k,KV,WRITE_SLOT,DIRECT)


@g.jit
def attention_body(Q,KV,Cache,Indices,Indptr,Sink,Max,Sum,Acc,Out,Counts,Counter,Health,M:gl.constexpr,T:gl.constexpr,SPLITS:gl.constexpr,BK:gl.constexpr=16):
 pid=gl.program_id(0)
 # Each row owns its compact cache. Direct consumption of current KV avoids
 # a producer/consumer barrier on the cache write.
 if pid<M:
  i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]))
  gl.store(Cache+pid*T*512+(T-1)*512+i,gl.load(KV+pid*512+i))
 # Bound tile prefetch by the actual compact-cache descriptor.
 PFSL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[16,128],[1,0])
 pfd=gl.amd.gfx1250.tdm.make_tensor_descriptor(Cache,[M*T,512],[512,1],[16,128],PFSL)
 for pf in range(pid,gl.cdiv(M*T,16)*4,256):
  gl.amd.gfx1250.tdm.prefetch(pfd,[(pf//4)*16,(pf%4)*128],pred=pf<gl.cdiv(M*T,16)*4,speculative=False)
 for task in range(pid,M*4*SPLITS,256):
  row=task%M;local=(task//M)%4*SPLITS+task//(M*4)
  compute_attention(Q+row*32768,KV+row*512,Cache+row*T*512,Indices+row*T,Indptr+row*2,Sink,
   Max+row*SPLITS*64,Sum+row*SPLITS*64,Acc+row*SPLITS*32768,Out+row*32768,Counts,
   T,T,SPLITS,16,BK,T-1,True,local)
 sync(Counter,Health,256,3)
 for task in range(pid,M*64,256):
  row=task//64;head=task%64
  _pa_decode_sparse_reduce(Max+row*SPLITS*64,Sum+row*SPLITS*64,Acc+row*SPLITS*32768,Sink,Indptr+row*2,Out+row*32768,
   SPLITS*64,64,1,SPLITS*64,64,1,SPLITS*64*512,64*512,512,1,64*512,512,1,
   64,512,SPLITS,1,512,BK,True,4,head)
