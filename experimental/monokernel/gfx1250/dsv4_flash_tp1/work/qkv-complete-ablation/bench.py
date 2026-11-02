from harness import *
out=root/'benchmark';out.mkdir(exist_ok=True)
runs=[]
for mode in ['separate','qa_fused','two','single']:
 for bn in ([64,128,256] if mode in ['two','single'] else [128]):
  s=State();s.run(mode,bn);torch.cuda.synchronize()
  o=s.obj;name=f'{mode}_bn{bn}'
  (out/(name+'.amdgcn')).write_text(o.asm['amdgcn'])
  resources=dict(vgpr=o.n_regs,spills=o.n_spills,lds=o.metadata.shared)
  cg=torch.cuda.CUDAGraph()
  with torch.cuda.graph(cg):
   for _ in range(128):s.run(mode,bn)
  for _ in range(5):cg.replay()
  torch.cuda.synchronize();assert all(v==0 if k=='health' else v for k,v in s.check_state().items())
  runs.append((name,s,cg,resources))
results=[]
for rnd in [1,2]:
 samples={n:[] for n,*_ in runs}
 for sample in range(40):
  for n,s,cg,res in (runs if sample%2==0 else reversed(runs)):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record();cg.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/128)
 row=dict(round=rnd,results={n:dict(median_us=statistics.median(samples[n]),samples_us=samples[n],resources=res,state=s.check_state()) for n,s,cg,res in runs})
 results.append(row);(out/'timings.json').write_text(json.dumps(results,indent=2));print(rnd,{n:v['median_us'] for n,v in row['results'].items()},flush=True)
