from common import *
from routing_batch import route_body as new
from routing_reference import route_body as old
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def run(LG,H,O,P,MP,GA,SC,GM,CH,WE,CNT,HEALTH,B,BIAS:gl.constexpr,NEW:gl.constexpr,M:gl.constexpr):
 if NEW:new(LG,H,O,P,MP,GA,SC,GM,B,BIAS,CH,WE,CNT,HEALTH,M)
 else:old(LG,H,O,P,MP,GA,SC,GM,B,BIAS,CH,WE,CNT,HEALTH,M)
def alloc(m):
 def z(s,dt=torch.int32):return torch.zeros(s,device='cuda',dtype=dt)
 return [z(256),z(256),z(1),z(m*6),z(m*6),z(m*6),z(m*6,torch.bfloat16),z((m,256)),z((m,256),torch.bfloat16),z(1),z(1)]
res=[]
for m in [1,2,4,8]:
 for seed in range(8):
  torch.manual_seed(seed);lg=torch.randn((m,256),device='cuda',dtype=torch.bfloat16);bias=torch.randn(256,device='cuda')*.1
  if seed==0:lg.zero_()
  a=alloc(m);b=alloc(m)
  for biased in [False,True]:
   run[(256,)](lg,*a,bias,biased,True,m,**kw);run[(256,)](lg,*b,bias,biased,False,m,**kw);torch.cuda.synchronize()
   checks=[torch.equal(u,v) for u,v in zip(a[:9],b[:9])]
   result=dict(m=m,seed=seed,bias=biased,equal=all(checks),fields=checks,health=int(a[-1]));res.append(result);assert all(checks) and not int(a[-1]),result
(ROOT/'routing-validation.json').write_text(json.dumps(res,indent=2));print('PASS',len(res),flush=True)
