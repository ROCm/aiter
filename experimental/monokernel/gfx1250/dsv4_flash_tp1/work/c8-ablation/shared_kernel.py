from shared_defs import *
@g.jit
def region(X,Q,S,G,LG,W,WS,GU,A,AQ,AS,W2,WS2,Y,COUNT,HEALTH,M:gl.constexpr):
 b.quant_body(X,Q,S,4096,128,8,M)
 b.router_body(X,G,LG,M)
 sync(COUNT,HEALTH,256,3)
 b.shared_body(Q,S,W,WS,GU,4096,64,M)
 sync(COUNT,HEALTH,256,3)
 b.activation_quant(GU,A,AQ,AS,M)
 sync(COUNT,HEALTH,256,3)
 b.shared_body(AQ,AS,W2,WS2,Y,2048,32,M)
