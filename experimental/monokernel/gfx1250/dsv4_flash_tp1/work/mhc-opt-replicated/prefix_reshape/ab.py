from setup import *
entries=[];resources={};checks=[]
ref=State();ref.run(R,'fused');torch.cuda.synchronize()
for mode in ['fused','l2','early','both']:
 s=State();s.run(R,mode);torch.cuda.synchronize();e=err(s.attn.out,ref.attn.out);assert e<.001 and s.check(),(mode,e)
 k=s.obj;resources[mode]=dict(vgpr=k.n_regs,spills=k.n_spills,lds=k.metadata.shared)
 (root/f'{mode}.amdgcn').write_text(k.asm['amdgcn']);(root/f'{mode}.llir').write_text(k.asm['llir'])
 print('ready',mode,resources[mode],flush=True)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(128):s.run(R,mode)
 for _ in range(5):g.replay()
 entries.append((mode,s,g));checks.append(dict(mode=mode,error=e))
rows=[]
for rnd in [1,2]:
 samples={n:[] for n,_,_ in entries}
 for i in range(40):
  for n,s,g in entries if i%2==0 else reversed(entries):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/128)
 row=dict(round=rnd,results={n:dict(median_us=statistics.median(samples[n]),samples_us=samples[n],health=s.check()) for n,s,_ in entries});rows.append(row);(root/'prefetch-timings.json').write_text(json.dumps(rows,indent=2));print(rnd,{n:v['median_us'] for n,v in row['results'].items()},flush=True)
(root/'prefetch-resources.json').write_text(json.dumps(resources,indent=2));(root/'prefetch-checks.json').write_text(json.dumps(checks,indent=2))
