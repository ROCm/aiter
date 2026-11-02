from common import *
from optimized_routing import route_body
from aiter.ops.triton.moe.moe_routing.routing import routing
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def run(LG,H,O,P,MP,GA,SC,GM,CH,WE,CNT,HEALTH,B,HAS:gl.constexpr,M:gl.constexpr):
 route_body(LG,H,O,P,MP,GA,SC,GM,B,HAS,CH,WE,CNT,HEALTH,M)
def alloc(m):
 z=lambda s,dt=torch.int32:torch.zeros(s,device='cuda',dtype=dt)
 return [z(256),z(256),z(1),z(m*6),z(m*6),z(m*6),z(m*6,torch.bfloat16),z((m+1,256)),z((m,256),torch.bfloat16),z(1),z(1)]
results=[]
for m in [8,16]:
 for seed in range(16):
  torch.manual_seed(seed);logits=torch.randn((m,256),device='cuda',dtype=torch.bfloat16);bias=torch.randn(256,device='cuda')*.1
  if seed==0:logits.zero_();bias.zero_()
  for biased in [False,True]:
   aa=alloc(m)
   rd,ga,sc=routing(logits,6,score_mode='sqrtsoftplus',bias=bias if biased else None,renorm=True,routed_scaling_factor=1.5)
   def fn():return run[(256,)](logits,*aa,bias,biased,m,**kw)
   fn();torch.cuda.synchronize()
   def check():return dict(hist=torch.equal(aa[0],rd.expt_data.hist),gather=torch.equal(aa[4].long(),ga.long()),scatter=torch.equal(aa[5].long(),sc.long()),scales=torch.equal(aa[6],rd.gate_scal),health=int(aa[-1]))
   first=check()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):
    for _ in range(8):fn()
   for _ in range(8):graph.replay()
   torch.cuda.synchronize();replay=check()
   row=dict(m=m,seed=seed,bias=biased,first=first,replay=replay);results.append(row)
   Path('/work/c16-native-parity/HERD_CHECKS.json').write_text(json.dumps(results,indent=2))
   assert all(first[k] and replay[k] for k in ['hist','gather','scatter','scales']) and not replay['health'],row
print('PASS',len(results),flush=True)
