from optimized_prefix import optimized_prefix
from combined import sync
from layout import unshuffle_body
from output_region import output_body
from post import post_body
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

@g.jit
def full_attention(RSh,RPlain,RFinal,HAF,HAS,HAB,AN,Mix,Temp,Raw,X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,QBW,QBS,QP,QC,QOut,Counter,Health,
 Cache,Indices,Indptr,Sink,M,L,A,O,Counts,
 OA,OAS,OB,OBS,OXQ,OXS,OAP,OAY,OAQ,OAScale,OBP,OY,
 T:gl.constexpr,PAGES:gl.constexpr,WRITE_SLOT:gl.constexpr,SPLITS:gl.constexpr,SHUFFLED_INPUT:gl.constexpr=True,DO_POST:gl.constexpr=True):
    if SHUFFLED_INPUT:
        unshuffle_body(RSh,RPlain)
        sync(Counter,Health,256,3)
    optimized_prefix(RPlain,HAF,HAS,HAB,AN,Mix,Temp,Raw,X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,QBW,QBS,QP,QC,QOut,Counter,Health,
     Cache,Indices,Indptr,Sink,M,L,A,O,Counts,T,PAGES,WRITE_SLOT,SPLITS,16,32,True,True,True)
    sync(Counter,Health,256,3)
    output_body(O,Cos,Sin,OA,OAS,OB,OBS,OXQ,OXS,OAP,OAY,OAQ,OAScale,OBP,OY,Counter,Health)
    if DO_POST:
        sync(Counter,Health,256,3)
        post_body(RPlain,OY,Mix,RFinal,128,False,2)
