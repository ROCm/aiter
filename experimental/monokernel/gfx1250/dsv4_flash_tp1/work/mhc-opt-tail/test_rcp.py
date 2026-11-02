from setup import *
from optimized_rcp import optimized
s=State();b=State()
def run(mode,r=R):
 if mode=='old':s.mhc_fused(r);return s.mobj
 if mode=='split':s.mhc_split(r);return None
 return optimized[(256,)](r,HAF,HAS,HAB,AN,s.mix,s.temp,s.raw,s.x,s.counter,s.health,MODE=mode,**kw)
for mode in [0,1]:
 for i in range(32):
  r=f2['inputs'][i%9].cuda().contiguous() if i<9 else torch.randn((4,4096),device='cuda').bfloat16()
  b.mhc_split(r);o=run(mode,r);torch.cuda.synchronize()
  e={k:err(getattr(s,k),getattr(b,k)) for k in ['mix','raw','x']};assert max(e.values())<.001,(mode,i,e)
 print('pass',mode,e,flush=True)
 (root/f'tail-rcp-{mode}.amdgcn').write_text(o.asm['amdgcn']);print(o.n_regs,o.n_spills,o.metadata.shared,flush=True)
graphs={}
for mode in ['split','old',0,1]:
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(128):run(mode)
 graphs[mode]=g
out=[]
for rnd in range(2):
 vals={str(k):[] for k in graphs}
 for j in range(40):
  for mode in list(graphs)[::1 if j%2 else -1]:
   g=graphs[mode];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record();g.replay();b.record();b.synchronize();vals[str(mode)].append(a.elapsed_time(b)*1000/128)
 row={k:statistics.median(v) for k,v in vals.items()};out.append(row);print(rnd,row,flush=True)
(root/'tail-rcp-timings.json').write_text(json.dumps(out,indent=2))
