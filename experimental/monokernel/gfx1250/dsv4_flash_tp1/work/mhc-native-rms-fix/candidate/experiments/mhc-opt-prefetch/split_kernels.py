from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store
from attention import prefetch_cache
from qb import project_qb
from stage import stage_qb
from aiter_adapt import _pa_decode_sparse,_pa_decode_sparse_reduce

@g.jit
def qa_reduce(P,Raw):
    group=gl.program_id(0)
    if group==0:
        i=gl.arange(0,1024,layout=gl.BlockedLayout([1],[32],[4],[0]));x=gl.full((1024,),0.,gl.float32,gl.BlockedLayout([1],[32],[4],[0]))
        for s in gl.static_range(16):x+=dev_load(P,s*1536+i)
        dev_store(x.to(gl.bfloat16).to(gl.float32),Raw,i)
    else:
        i2=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]));x2=gl.full((512,),0.,gl.float32,gl.BlockedLayout([1],[32],[4],[0]))
        for s in gl.static_range(16):x2+=dev_load(P,s*1536+1024+i2)
        dev_store(x2.to(gl.bfloat16).to(gl.float32),Raw,1024+i2)

@g.jit
def qa_norm(Raw,NW,Y):
    if gl.program_id(0)==0:
        i=gl.arange(0,1024,layout=gl.BlockedLayout([1],[32],[4],[0]));x=dev_load(Raw,i)
        y=x*gl.rsqrt(gl.sum(x*x,0)/1024+1.e-6)*dev_load(NW,i);dev_store(y,Y,i)
    else:
        i2=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]));x2=dev_load(Raw,1024+i2)
        y2=x2*gl.rsqrt(gl.sum(x2*x2,0)/512+1.e-6)*dev_load(NW,1024+i2);dev_store(y2,Y,1024+i2)

@g.jit
def qb_stage(X,XS,W,WS,P,Count,Cos,Sin,O):
    pid=gl.program_id(0)
    SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[128,128],[1,0])
    shared=gl.allocate_shared_memory(W.dtype.element_ty,[2,128,128],SL)
    desc=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=W+pid*131072,shape=[128,1024],strides=[1024,1],block_shape=[128,128],layout=SL)
    stage_qb(W,pid,desc,shared,2,True)
    project_qb(X,XS,W,WS,P,Count,Cos,Sin,O,pid,False,128,1,True,4,128,4,True,True,'.ca',True,False,TDM=True,FirstTile=shared,Desc=desc)

@g.jit
def cache_store(KV,Cache,SLOT:gl.constexpr):
    i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]));dev_store(dev_load(KV,i),Cache,SLOT*512+i)

@g.jit
def attention_partials(Q,Cache,Indices,Indptr,Sink,M,L,A,O,PAGES:gl.constexpr,SPLITS:gl.constexpr):
    prefetch_cache(Cache,Indices,PAGES,4*SPLITS)
    pid=gl.program_id(0);h=pid//SPLITS;k=pid%SPLITS
    _pa_decode_sparse(Q,Cache,Cache,Indices,Indptr,M,L,A,Sink,O,PAGES,
        64*512,512,1,512,1,1,SPLITS*64,64,1,SPLITS*64,64,1,
        SPLITS*64*512,64*512,512,1,64*512,512,1,
        64,512,SPLITS,0.04419417382415922,16,512,32,True,False,128,4,True,4,h,k)

@g.jit
def attention_reduce(M,L,A,Sink,Indptr,O,SPLITS:gl.constexpr):
    _pa_decode_sparse_reduce(M,L,A,Sink,Indptr,O,
        SPLITS*64,64,1,SPLITS*64,64,1,
        SPLITS*64*512,64*512,512,1,64*512,512,1,
        64,512,SPLITS,16,512,32,True,4,gl.program_id(0))
