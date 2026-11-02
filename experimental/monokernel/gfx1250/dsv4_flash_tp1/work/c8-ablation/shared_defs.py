import sys,torch,json
sys.path[:0]=['/work/batch-full','/work/modelwide','/work/full-attention-native/prefix','/work/ffn-c1/moe']
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import ffn_batch as b, ffn_precise as c
from combined import sync
@g.jit
def batch(X,Q,S,MX,MS,G,LG,W,WS,GU,A,AQ,AS,W2,WS2,Y,H,O,P,MP,GA,SC,GM,BI,CH,WE,COUNT,HEALTH,M:gl.constexpr):
 b.quant_body(X,Q,S,4096,128,8,M);b.quant_body(X,MX,MS,4096,32,1,M);b.router_body(X,G,LG,M)
 sync(COUNT,HEALTH,256,3)
 b.route_body(LG,H,O,P,MP,GA,SC,GM,BI,True,CH,WE,COUNT,HEALTH,M)
 b.shared_body(Q,S,W,WS,GU,4096,64,M)
 sync(COUNT,HEALTH,256,3)
 b.activation_quant(GU,A,AQ,AS,M)
 sync(COUNT,HEALTH,256,3)
 b.shared_body(AQ,AS,W2,WS2,Y,2048,32,M)
@g.jit
def single(X,Q,S,MX,MS,G,LG,W,WS,GU,A,AQ,AS,W2,WS2,Y,H,O,P,MP,GA,SC,GM,BI,COUNT,HEALTH):
 c.quant_body(X,Q,S,4096,128,8);c.quant_body(X,MX,MS,4096,32);c.router_body(X,G,LG)
 sync(COUNT,HEALTH,256,3)
 c.route_body(LG,H,O,P,MP,GA,SC,GM,BI,True)
 c.shared_body(Q,S,W,WS,GU,4096,64)
 sync(COUNT,HEALTH,256,3)
 c.activation_quant(GU,A,AQ,AS)
 sync(COUNT,HEALTH,256,3)
 c.shared_body(AQ,AS,W2,WS2,Y,2048,32)
def e(sh,dt=torch.bfloat16):return torch.empty(sh,device='cuda',dtype=dt)
def z(sh,dt=torch.int32):return torch.zeros(sh,device='cuda',dtype=dt)
def scratch(m):return [e((m,4096),torch.float8_e4m3fn),z((32,8),torch.uint8),e((m,4096),torch.float8_e4m3fn),e((m,128),torch.uint8),e((m,256)),e((m,4096)),e((m,2048)),e((m,2048),torch.float8_e4m3fn),z((16,8),torch.uint8),e((m,4096)),z(256),z(256),z(1),z(m*6),z(m*6),z(m*6),e(m*6),z((m,256)),e((m,256)),z(1),z(1)]
