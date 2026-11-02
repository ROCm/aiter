from setup import *
entries=[];resources={}
for t in [128,1024]:
 for mode in ['unfused','three_regions','fused']:
  s=State(t);s.run(R,mode);torch.cuda.synchronize();name=f't{t}_{mode}'
  if mode=='fused':
   k=s.obj;resources[name]=dict(vgpr=k.n_regs,lds=k.metadata.shared,spills=k.n_spills)
   (root/f'{name}.amdgcn').write_text(k.asm['amdgcn']);(root/f'{name}.llir').write_text(k.asm['llir'])
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for _ in range(128):s.run(R,mode)
  for _ in range(5):g.replay()
  entries.append((name,s,g));print('ready',name,flush=True)
# Fresh per-group comparisons, same checkpoint-derived inputs; operation inputs
# are not modified in place, so repeated calls do not require restore kernels.
for group in ['mhc','qkv','attn']:
 for mode in ['split','fused']:
  s=State();s.run(R,'unfused');fn=getattr(s,group+'_'+mode);args=(R,) if group=='mhc' else ()
  fn(*args);torch.cuda.synchronize();name=f'group_{group}_{mode}'
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for _ in range(128):fn(*args)
  for _ in range(5):g.replay()
  entries.append((name,s,g));print('ready',name,flush=True)
results=[]
for rnd in [1,2]:
 samples={n:[] for n,_,_ in entries}
 for i in range(40):
  for n,s,g in entries if i%2==0 else reversed(entries):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/128)
 row=dict(round=rnd,results={n:dict(median_us=statistics.median(samples[n]),samples_us=samples[n],health=s.check()) for n,s,_ in entries});results.append(row)
 (root/'timings.json').write_text(json.dumps(results,indent=2));print(rnd,{n:v['median_us'] for n,v in row['results'].items()},flush=True)
(root/'resources.json').write_text(json.dumps(resources,indent=2))
(root/'environment.json').write_text(json.dumps(dict(gpu=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,triton=__import__('triton').__version__),indent=2))
