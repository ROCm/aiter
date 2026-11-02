from setup import *
entries=[];resources={}
modes=['unfused','fused','opt','opt_l2','opt_both']
for t in [128,1024]:
 ref=State(t);ref.run(R,'fused');torch.cuda.synchronize();expected=ref.attn.out.clone()
 for mode in modes:
  s=State(t);s.run(R,mode);torch.cuda.synchronize();e=err(s.attn.out,expected);assert e<.001 and s.check(),(t,mode,e)
  name=f't{t}_{mode}'
  if mode!='unfused':
   k=s.obj;resources[name]=dict(vgpr=k.n_regs,lds=k.metadata.shared,spills=k.n_spills)
   (root/f'{name}.amdgcn').write_text(k.asm['amdgcn']);(root/f'{name}.llir').write_text(k.asm['llir'])
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for _ in range(128):s.run(R,mode)
  for _ in range(5):g.replay()
  entries.append((name,s,g));print('ready',name,flush=True)
rows=[]
for rnd in [1,2]:
 samples={n:[] for n,_,_ in entries}
 for i in range(40):
  for n,s,g in entries if i%2==0 else reversed(entries):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/128)
 row=dict(round=rnd,results={n:dict(median_us=statistics.median(samples[n]),samples_us=samples[n],health=s.check()) for n,s,_ in entries});rows.append(row);(root/'prefix-timings.json').write_text(json.dumps(rows,indent=2));print(rnd,{n:v['median_us'] for n,v in row['results'].items()},flush=True)
(root/'prefix-resources.json').write_text(json.dumps(resources,indent=2))
