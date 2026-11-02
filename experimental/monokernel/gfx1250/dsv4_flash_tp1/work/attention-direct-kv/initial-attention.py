from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from aiter_adapt import _pa_decode_sparse, _pa_decode_sparse_reduce
from device_scope_helpers import dev_load,dev_store
from combined import sync,complete
from vector_original import attention as vector_attention

@g.jit
def prefetch_cache(Cache,Indices,T:gl.constexpr):
    pid=gl.program_id(0)
    if pid<gl.cdiv(T,32):
        tid=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
        row=pid*32+tid//4
        slot=dev_load(Indices,row,row<T,-1)
        addr=(Cache+gl.maximum(slot,0)*512+(tid%4)*128).to(gl.uint64)
        gl.inline_asm_elementwise('global_prefetch_b8 $1, off scope:SCOPE_DEV th:TH_LOAD_NT\n v_mov_b32 $0,0',constraints='=v,v,~{memory}',args=[addr],dtype=gl.int32,is_pure=False,pack=1)

@g.jit
def compute_attention(Q,Cache,Indices,Indptr,Sink,M,L,A,O,Counts,T:gl.constexpr,PAGES:gl.constexpr,SPLITS:gl.constexpr,BH:gl.constexpr=16,BK:gl.constexpr=32):
    pid=gl.program_id(0)
    if pid<(64//BH)*SPLITS:
        h=pid//SPLITS;k=pid%SPLITS
        _pa_decode_sparse(Q,Cache,Cache,Indices,Indptr,M,L,A,Sink,O,PAGES,
            64*512,512,1,512,1,1,
            SPLITS*64,64,1,SPLITS*64,64,1,
            SPLITS*64*512,64*512,512,1,64*512,512,1,
            64,512,SPLITS,0.04419417382415922,BH,512,BK,True,False,128,4,True,4,h,k)
        if SPLITS>1:
            gl.inline_asm_elementwise('s_wait_loadcnt 0\n s_wait_storecnt 0\n s_mov_b32 $0,0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
            gl.barrier()
            ticket=gl.atomic_add(Counts+h,1,sem='relaxed',scope='gpu')
            if ticket==SPLITS-1:
                _pa_decode_sparse_reduce(M,L,A,Sink,Indptr,O,
                    SPLITS*64,64,1,SPLITS*64,64,1,
                    SPLITS*64*512,64*512,512,1,64*512,512,1,
                    64,512,SPLITS,BH,512,BK,True,4,h)
                gl.atomic_xchg(Counts+h,0,sem='relaxed',scope='gpu')

@g.jit
def cache_attention_body(Q,KV,Cache,Indices,Indptr,Sink,M,L,A,O,Counts,Counter,Health,T:gl.constexpr,PAGES:gl.constexpr,WRITE_SLOT:gl.constexpr,SPLITS:gl.constexpr,GRID:gl.constexpr,BH:gl.constexpr=16,BK:gl.constexpr=32,VECTOR:gl.constexpr=False):
    if gl.program_id(0)==0:
        i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]))
        dev_store(dev_load(KV,i),Cache,WRITE_SLOT*512+i)
    sync(Counter,Health,GRID,3)
    if VECTOR:
        vector_attention(Q,KV,Cache,Sink,O,GRID)
    else:
        prefetch_cache(Cache,Indices,T)
        compute_attention(Q,Cache,Indices,Indptr,Sink,M,L,A,O,Counts,T,PAGES,SPLITS,BH,BK)

@g.jit
def attention_kernel(Q,KV,Cache,Indices,Indptr,Sink,M,L,A,O,Counts,Counter,Health,T:gl.constexpr,PAGES:gl.constexpr,WRITE_SLOT:gl.constexpr,SPLITS:gl.constexpr,GRID:gl.constexpr,BH:gl.constexpr=16,BK:gl.constexpr=32,VECTOR:gl.constexpr=False):
    cache_attention_body(Q,KV,Cache,Indices,Indptr,Sink,M,L,A,O,Counts,Counter,Health,T,PAGES,WRITE_SLOT,SPLITS,GRID,BH,BK,VECTOR)

@g.jit
def qkv_attention(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,QBW,QBS,QP,QC,QOut,Counter,Health,
                  Cache,Indices,Indptr,Sink,M,L,A,O,Counts,T:gl.constexpr,PAGES:gl.constexpr,WRITE_SLOT:gl.constexpr,SPLITS:gl.constexpr,BH:gl.constexpr=16,BK:gl.constexpr=32):
    complete(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,QBW,QBS,QP,QC,QOut,Counter,Health,BN=128,GRID=256,STYLE=3,U=4,QAMODE=0,ALLTDM=True,EARLY=True,PREFETCH=2)
    sync(Counter,Health,256,3)
    cache_attention_body(QOut,KVOut,Cache,Indices,Indptr,Sink,M,L,A,O,Counts,Counter,Health,T,PAGES,WRITE_SLOT,SPLITS,256,BH,BK)
