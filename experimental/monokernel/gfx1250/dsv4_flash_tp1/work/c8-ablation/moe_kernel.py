from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from ffn_batch import quant_body
from moe_multitoken_prototype import fused
from combined_scaled import sync
@g.jit
def moe_region(X,XQ,XS,A,Counter,Health):
 quant_body(X,XQ,XS,4096,32,1,8)
 sync(Counter,Health,256,3)
 fused(XQ,XS,*A,GRID=256,BUFFERS=3)
