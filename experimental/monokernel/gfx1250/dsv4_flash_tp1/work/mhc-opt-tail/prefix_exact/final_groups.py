from setup import *
from tdm_tail import candidate
s=State()
def run(mode):
 if mode=='old':s.mhc_fused(R);return s.mobj
 if mode=='split':s.mhc_split(R);return None
 return candidate[(256,)](R,HAF,HAS,HAB,AN,s.mix,s.temp,s.raw,s.x,s.counter,s.health,EARLY=True,**kw)
graphs={};resources={}
for mode in ['split','old','optimized']:
 for _ in range(3):o=run(mode)
 if o is not None:
  resources[mode]={'regs':o.n_regs,'spills':o.n_spills,'shared':o.metadata.shared};(root/f'group-final-{mode}.amdgcn').write_text(o.asm['amdgcn'])
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(128):run(mode)
 graphs[mode]=g
out=[]
for rnd in range(2):
 vals={k:[] for k in graphs}
 for j in range(60):
  for mode in list(graphs)[::1 if j%2 else -1]:
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();graphs[mode].replay();b.record();b.synchronize();vals[mode].append(a.elapsed_time(b)*1000/128)
 row={k:statistics.median(v) for k,v in vals.items()};out.append(row);print(rnd,row,flush=True)
(root/'final-group-timings.json').write_text(json.dumps({'rounds':out,'resources':resources},indent=2))
