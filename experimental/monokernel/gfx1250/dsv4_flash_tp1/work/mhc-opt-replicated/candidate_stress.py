from setup import *
from candidate import candidate
s=State();b=State();xs=[x.cuda().contiguous() for x in f2['inputs']]
for i in range(23):
 gen=torch.Generator(device='cuda').manual_seed(9991+i);xs.append((torch.randn((4,4096),device='cuda',generator=gen)*2**(i%7-3)).bfloat16())
refs=[]
for x in xs:
 b.mhc_split(x);refs.append((b.mix.clone(),b.raw.clone(),b.x.clone()))
r=R.clone();om=torch.empty((32,24),device='cuda');orr=torch.empty((32,4096),device='cuda',dtype=torch.bfloat16);ox=torch.empty_like(orr)
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
 for i,x in enumerate(xs):
  r.copy_(x);s.mix.fill_(float('nan'));s.raw.fill_(float('nan'));s.x.fill_(float('nan'))
  candidate[(256,)](r,HAF,HAS,HAB,AN,s.mix,s.temp,s.raw,s.x,s.counter,s.health,EARLY=True,**kw)
  om[i].copy_(s.mix);orr[i].copy_(s.raw);ox[i].copy_(s.x)
rm=torch.stack([v[0] for v in refs]);rr=torch.stack([v[1] for v in refs]);rx=torch.stack([v[2] for v in refs]);mx=0
for i in range(320):
 g.replay();es=[err(a,b) for a,b in [(om,rm),(orr,rr),(ox,rx)]];assert max(es)<.001 and s.health.item()==0,es;mx=max(mx,max(es))
(root/'candidate-stress.json').write_text(json.dumps({'checks':10240,'max_error':mx,'health':s.health.item()},indent=2));print('stress',mx,flush=True)
