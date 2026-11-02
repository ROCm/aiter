from previous.common import *
from previous.optimized_routing import route_body
from aiter.ops.triton.moe.moe_routing.routing import routing
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def run(LG,H,O,P,MP,GA,SC,GM,CH,WE,CNT,HEALTH,M:gl.constexpr):
 route_body(LG,H,O,P,MP,GA,SC,GM,None,False,CH,WE,CNT,HEALTH,M)
def z(s,dt=torch.int32):return torch.zeros(s,device='cuda',dtype=dt)
torch.manual_seed(6764);logits=torch.randn((M,256),device='cuda',dtype=torch.bfloat16)
h=z(256);o=z(256);p=z(1);mp=z(M*6);ga=z(M*6);sc=z(M*6);gm=z(M*6,torch.bfloat16);ch=z((M+1,256));we=z((M,256),torch.bfloat16);cnt=z(1);health=z(1)
def fused():return run[(256,)](logits,h,o,p,mp,ga,sc,gm,ch,we,cnt,health,M,**kw)
def native():return routing(logits,6,score_mode='sqrtsoftplus',renorm=True,routed_scaling_factor=1.5)
rd,gather,scatter=native();fused();torch.cuda.synchronize()
bench('routing',{'native':native,'fused':fused},dict(hist_equal=torch.equal(h,rd.expt_data.hist),gather_equal=torch.equal(ga.long(),gather.long()),scatter_equal=torch.equal(sc.long(),scatter.long()),scales=err(gm,rd.gate_scal),health=int(health),note='native HERD enabled at C16; excludes router GEMM'))
