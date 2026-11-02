from attention import *
from mhc import mhc_body

@g.jit
def full_prefix(R,HAF,HAS,HAB,AN,Mix,Temp,Raw,X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,QBW,QBS,QP,QC,QOut,Counter,Health,
                  Cache,Indices,Indptr,Sink,M,L,A,O,Counts,T:gl.constexpr,PAGES:gl.constexpr,WRITE_SLOT:gl.constexpr,SPLITS:gl.constexpr,BH:gl.constexpr=16,BK:gl.constexpr=32,DIRECT:gl.constexpr=True):
    mhc_body(R,HAF,HAS,HAB,AN,Mix,Temp,Raw,X,Counter,Health)
    sync(Counter,Health,256,3)
    complete(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,QBW,QBS,QP,QC,QOut,Counter,Health,BN=128,GRID=256,STYLE=3,U=4,QAMODE=0,ALLTDM=True,EARLY=True,PREFETCH=2)
    sync(Counter,Health,256,3)
    cache_attention_body(QOut,KVOut,Cache,Indices,Indptr,Sink,M,L,A,O,Counts,Counter,Health,T,PAGES,WRITE_SLOT,SPLITS,256,BH,BK,False,DIRECT)
