import sys,json,os
sys.path[:0]=['/app/aiter',os.path.dirname(__file__)]
import torch
from routing import route_body
from aiter.ops.triton.moe.moe_routing.routing import routing
z=lambda n,d:torch.zeros(n,device='cuda',dtype=d)
hist=z(256,torch.int32);offs=hist.clone();pad=z(1,torch.int32)
ids=z(6,torch.int32);gather=ids.clone();scatter=ids.clone();gamma=z(6,torch.bfloat16)
checks=[]
for seed in range(100):
 torch.manual_seed(seed)
 x=torch.randn(1,256,device='cuda',dtype=torch.bfloat16)* (0.001 if seed%4==0 else 2)
 bias=torch.randn(256,device='cuda')*.2
 for has in [False,True]:
  ref,_,_=routing(x,6,score_mode='sqrtsoftplus',bias=bias if has else None,renorm=True,routed_scaling_factor=1.5)
  route_body[(1,)](x,hist,offs,pad,ids,gather,scatter,gamma,bias,has,num_warps=4)
  checks.append(dict(seed=seed,bias=has,ids=torch.equal(ids,ref.topk_ids.flatten()),weights=torch.equal(gamma,ref.gate_scal.flatten()),max_abs=(gamma.float()-ref.gate_scal.flatten().float()).abs().max().item()))
print(json.dumps(dict(cases=len(checks),ids_all=all(x['ids'] for x in checks),weights_all=all(x['weights'] for x in checks),max_abs=max(x['max_abs'] for x in checks))),flush=True)
open('/work/modelwide/routing-validation.json','w').write(json.dumps(checks,indent=2))
