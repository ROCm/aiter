from setup import *
from candidate import candidate
entries=[]
for name in ['split','old','late','early','poll']:
 s=State()
 def run(r,s=s,name=name):
  if name=='split':s.mhc_split(r)
  elif name=='old':s.mhc_fused(r)
  else:s.mobj=candidate[(256,)](r,HAF,HAS,HAB,AN,s.mix,s.temp,s.raw,s.x,s.counter,s.health,EARLY=2 if name=='poll' else name=='early',**kw)
 run(R);torch.cuda.synchronize();entries.append((name,s,run))
 print('ready',name,flush=True)
 if name in ['early','late']:
  (root/(name+'.amdgcn')).write_text(s.mobj.asm['amdgcn'])
  print(s.mobj.n_regs,s.mobj.metadata.shared,s.mobj.n_spills,flush=True)
checks=[]
for j in range(64):
 r=f2['inputs'][j%9].cuda().clone()
 if j>=9:r=(r.float()+torch.randn_like(r.float())*.01*(1+j%4)).bfloat16()
 for name,s,run in entries:run(r)
 torch.cuda.synchronize();ref=entries[0][1]
 for name,s,run in entries[1:]:
  es={key:err(getattr(s,key),getattr(ref,key)) for key in ['mix','raw','x']};checks.append([j,name,es]);assert max(es.values())<.001,(j,name,es)
(root/'candidate-validation.json').write_text(json.dumps(checks,indent=2))
graphs=[]
for name,s,run in entries:
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(128):run(R)
 for _ in range(5):g.replay()
 graphs.append((name,s,g))
results=[]
for rnd in [1,2]:
 samples={n:[] for n,s,g in graphs}
 for j in range(40):
  for n,s,g in graphs if j%2==0 else reversed(graphs):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/128)
 results.append({'round':rnd,'us':{n:statistics.median(v) for n,v in samples.items()},'samples':samples});print(results[-1]['us'],flush=True)
(root/'candidate-timings.json').write_text(json.dumps(results,indent=2))
