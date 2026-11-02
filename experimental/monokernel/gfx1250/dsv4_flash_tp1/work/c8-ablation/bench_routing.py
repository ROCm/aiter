from common import *
from routing_batch import route_body
from aiter.ops.triton.moe.moe_routing.routing import routing
from triton.experimental import gluon as g
@g.jit
def run(LG,H,O,P,MP,GA,SC,GM,CH,WE,CNT,HEALTH):
 route_body(LG,H,O,P,MP,GA,SC,GM,None,False,CH,WE,CNT,HEALTH,8)
def z(s,dt=torch.int32):return torch.zeros(s,device='cuda',dtype=dt)
torch.manual_seed(6764);logits=torch.randn((8,256),device='cuda',dtype=torch.bfloat16)
h=z(256);o=z(256);p=z(1);mp=z(48);ga=z(48);sc=z(48);gm=z(48,torch.bfloat16);ch=z((8,256));we=z((8,256),torch.bfloat16);cnt=z(1);health=z(1)
def fused():return run[(256,)](logits,h,o,p,mp,ga,sc,gm,ch,we,cnt,health,**kw)
def native():return routing(logits,6,score_mode='sqrtsoftplus',renorm=True,routed_scaling_factor=1.5)
rd,gather,scatter=native();fused();torch.cuda.synchronize()
bench('routing',{'native':native,'fused':fused},dict(hist_equal=torch.equal(h,rd.expt_data.hist),health=int(health),note='learned top6 sqrtsoftplus routing, no bias; excludes router GEMM'))
