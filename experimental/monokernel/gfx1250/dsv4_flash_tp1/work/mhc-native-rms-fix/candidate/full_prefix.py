from attention import *
from mhc import mhc_body

@g.jit
def full_prefix(R,HAF,HAS,HAB,AN,Mix,Temp,Raw,X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,QBW,QBS,QP,QC,QOut,Counter,Health,
                  Cache,Indices,Indptr,Sink,M,L,A,O,Counts,T:gl.constexpr,PAGES:gl.constexpr,WRITE_SLOT:gl.constexpr,SPLITS:gl.constexpr,BH:gl.constexpr=16,BK:gl.constexpr=32,DIRECT:gl.constexpr=True,EARLY_QA:gl.constexpr=False,L2_QA:gl.constexpr=False):
    pid=gl.program_id(0)
    if L2_QA and pid<192:
        tid=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
        addr=(W+(pid//16)*128*4096+(pid%16)*256+tid*4096).to(gl.uint64)
        gl.inline_asm_elementwise('global_prefetch_b8 $1, off scope:SCOPE_DEV th:TH_LOAD_NT\n v_mov_b32 $0,0',constraints='=v,v,~{memory}',args=[addr],dtype=gl.int32,is_pure=False,pack=1)
    if EARLY_QA:
        QSL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[128,128],[1,0])
        qsm=gl.allocate_shared_memory(W.dtype.element_ty,[2,128,128],QSL)
        qdesc=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=W+(pid//16)*128*4096+(pid%16)*256,shape=[128,256],strides=[4096,1],block_shape=[128,128],layout=QSL)
        if pid<192:
            gl.amd.gfx1250.tdm.async_load(qdesc,[0,0],qsm.index(0))
            gl.amd.gfx1250.tdm.async_load(qdesc,[0,128],qsm.index(1))
    else:
        qsm=None;qdesc=None
    mhc_body(R,HAF,HAS,HAB,AN,Mix,Temp,Raw,X,Counter,Health)
    sync(Counter,Health,256,3)
    complete(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,QBW,QBS,QP,QC,QOut,Counter,Health,BN=128,GRID=256,STYLE=3,U=4,QAMODE=0,ALLTDM=True,EARLY=True,PREFETCH=2,QAPreShared=qsm,QAPreDesc=qdesc)
    sync(Counter,Health,256,3)
    cache_attention_body(QOut,KVOut,Cache,Indices,Indptr,Sink,M,L,A,O,Counts,Counter,Health,T,PAGES,WRITE_SLOT,SPLITS,256,BH,BK,False,DIRECT)
