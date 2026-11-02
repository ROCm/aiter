from setup import *
from optimized_rcp import optimized
import aiter
s=State()
def native(r=R,normfused=False):
 p,c,y=aiter.mhc_pre(r.view(1,4,4096),HAF,HAS,HAB,hc_post_mult_value=2.,norm_weight=AN if normfused else None)
 if not normfused:
  from mhc import norm
  norm[(1,)](y,AN,s.x,4096,1,True,1,**kw)
 return p,c,y if normfused else s.x
errs=[]
for i in range(32):
 r=f2['inputs'][i%9].cuda().contiguous() if i<9 else torch.randn((4,4096),device='cuda').bfloat16()
 s.mhc_split(r);ref=s.x.clone();mix=s.mix.clone()
 for nf in [False,True]:
  p,c,x=native(r,nf);es={'post':err(p.flatten(),mix[4:8]),'comb':err(c.flatten(),mix[8:]),'x':err(x.flatten(),ref)}
  errs.append({'case':i,'normfused':nf,'errors':es})
  # Native diagnostic: keep accuracy differences visible.
print('errors',max(v['errors']['x'] for v in errs if not v['normfused']),max(v['errors']['x'] for v in errs if v['normfused']),flush=True)
funcs={'native_preserve_bf16':lambda:native(),'native_fused_norm':lambda:native(normfused=True),'optimized':lambda:optimized[(256,)](R,HAF,HAS,HAB,AN,s.mix,s.temp,s.raw,s.x,s.counter,s.health,MODE=1,**kw)}
graphs={}
for k,fn in funcs.items():
 for _ in range(3):fn()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(128):fn()
 graphs[k]=g
out=[]
for rnd in range(2):
 vals={k:[] for k in graphs}
 for j in range(40):
  for k in list(graphs)[::1 if j%2 else -1]:
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();graphs[k].replay();b.record();b.synchronize();vals[k].append(a.elapsed_time(b)*1000/128)
 row={k:statistics.median(v) for k,v in vals.items()};out.append(row);print(rnd,row,flush=True)
(root/'native-timings.json').write_text(json.dumps({'timings':out,'errors':errs},indent=2))
