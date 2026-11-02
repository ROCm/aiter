from setup import *
from tdm_tail import candidate
weights=[HAF.clone() for _ in range(256)]
entries=[];resources={}
for style in ['reused','rotating256']:
 for mode in ['split','old','optimized']:
  s=State()
  def run(w):
   if mode=='split':
    for stage,grid in enumerate([256,24,1,16,1]):mhc_stage[(grid,)](R,w,HAS,HAB,AN,s.mix,s.temp,s.raw,s.x,STAGE=stage,**kw)
   elif mode=='old':mhc_region[(256,)](R,w,HAS,HAB,AN,s.mix,s.temp,s.raw,s.x,s.counter,s.health,**kw)
   else:candidate[(256,)](R,w,HAS,HAB,AN,s.mix,s.temp,s.raw,s.x,s.counter,s.health,EARLY=True,**kw)
  run(HAF);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for w in weights:run(w if style=='rotating256' else HAF)
  for _ in range(5):g.replay()
  entries.append((style+'_'+mode,s,g));print('ready',style,mode,flush=True)
results=[]
for rnd in [1,2]:
 samples={n:[] for n,s,g in entries}
 for j in range(40):
  for n,s,g in entries if j%2==0 else reversed(entries):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/256)
 row={'round':rnd,'us':{n:statistics.median(v) for n,v in samples.items()},'samples_us':samples};results.append(row);print(row['us'],flush=True)
(root/'rotating-mhc-timings.json').write_text(json.dumps(results,indent=2))
