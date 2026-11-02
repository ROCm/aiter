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
        else:
            LH:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
            hh=h*BH+gl.arange(0,BH,layout=LH)
            gl.store(M+k*64+hh,float('-inf'))
            gl.store(L+k*64+hh,0.)
            LA:gl.constexpr=gl.BlockedLayout([1,4],[4,8],[4,1],[1,0])
            hi=h*BH+gl.arange(0,BH,layout=gl.SliceLayout(1,LA))
            di=gl.arange(0,512,layout=gl.SliceLayout(0,LA))
            gl.store(A+k*64*512+hi[:,None]*512+di[None,:],0.)


@g.jit
def attention_body(Q,KV,Cache,Indices,Indptr,Sink,Max,Sum,Acc,Out,Counts,Counter,Health,M:gl.constexpr,T:gl.constexpr,SPLITS:gl.constexpr,BK:gl.constexpr=16):
 pid=gl.program_id(0)
 # Each row owns its compact cache. Direct consumption of current KV avoids
 # a producer/consumer barrier on the cache write.
 if pid<M:
  i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]))
  gl.store(Cache+pid*T*512+(T-1)*512+i,gl.load(KV+pid*512+i))
 # Mask every prefetch lane against the actual allocated cache extent.
 tid=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
 for line in range(pid*128,gl.cdiv(M*T*1024,256),256*128):
  masked_line_prefetch(Cache,(line+tid)*256,M*T*1024)
 for task in range(pid,M*4*SPLITS,256):
  row=task//(4*SPLITS);local=task%(4*SPLITS)
  compute_attention(Q+row*32768,KV+row*512,Cache+row*T*512,Indices+row*T,Indptr+row*2,Sink,
   Max+row*SPLITS*64,Sum+row*SPLITS*64,Acc+row*SPLITS*32768,Out+row*32768,Counts,
   T,T,SPLITS,16,BK,T-1,True,local)
 sync(Counter,Health,256,3)
 for task in range(pid,M*64,256):
  row=task//64;head=task%64
  _pa_decode_sparse_reduce(Max+row*SPLITS*64,Sum+row*SPLITS*64,Acc+row*SPLITS*32768,Sink,Indptr+row*2,Out+row*32768,
   SPLITS*64,64,1,SPLITS*64,64,1,SPLITS*64*512,64*512,512,1,64*512,512,1,
   64,512,SPLITS,1,512,BK,True,4,head)
