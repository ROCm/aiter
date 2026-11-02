from setup import *
from optimized_exact import optimized
s=State();b=State();mx={k:0 for k in ['mix','raw','x']};neq={k:0 for k in mx}
for i in range(512):
 gen=torch.Generator(device='cuda').manual_seed(71100+i);r=(torch.randn((4,4096),device='cuda',generator=gen)*2**(i%15-7)).bfloat16()
 b.mhc_split(r);optimized[(256,)](r,HAF,HAS,HAB,AN,s.mix,s.temp,s.raw,s.x,s.counter,s.health,MODE=1,**kw)
 for k in mx:
  mx[k]=max(mx[k],err(getattr(s,k),getattr(b,k)));neq[k]+=int(not torch.equal(getattr(s,k)[:4] if k=='mix' else getattr(s,k),getattr(b,k)[:4] if k=='mix' else getattr(b,k)))
 if i==123:print('case132',mx,neq,flush=True)
print('all512',mx,neq,flush=True)
(root/'exact-check.json').write_text(json.dumps({'max_error':mx,'non_bitwise_equal':neq},indent=2))
