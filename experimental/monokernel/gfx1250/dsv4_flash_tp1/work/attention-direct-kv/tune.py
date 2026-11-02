from common import *
results=[];resources={}
for t in [128,1024]:
 idx=torch.arange(t,device='cuda',dtype=torch.int32);cache=makecache(t);cache[-1].copy_(KV);ref=reference(Q,cache,idx,Sink);entries=[]
 for bh,s in [(16,1),(16,2),(16,4),(32,1),(32,2),(64,1)]:
  name=f'h{bh}_s{s}';obj=AttentionRegion(cache.clone(),idx,Sink,splits=s,block_h=bh);obj(Q,KV);torch.cuda.synchronize();k=obj.obj
  e=error(obj.out,ref);assert e<.001 and obj.check(),(name,e)
  resources[f'{t}_{name}']=dict(vgpr=k.n_regs,lds=k.metadata.shared,spills=k.n_spills,error=e)
  (root/f'tune-{t}-{name}.amdgcn').write_text(k.asm['amdgcn'])
  if k.n_spills:continue
  cg=torch.cuda.CUDAGraph()
  with torch.cuda.graph(cg):
   for _ in range(128):obj(Q,KV)
  for _ in range(5):cg.replay()
  entries.append((name,obj,cg))
 for rnd in [1,2]:
  samples={n:[] for n,_,_ in entries}
  for i in range(40):
   for n,obj,cg in entries if i%2==0 else reversed(entries):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();cg.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/128)
  row=dict(tokens=t,round=rnd,results={n:dict(median_us=statistics.median(samples[n]),samples_us=samples[n],health=obj.check()) for n,obj,_ in entries});results.append(row)
  (root/'tuning.json').write_text(json.dumps(results,indent=2));print(t,rnd,{n:v['median_us'] for n,v in row['results'].items()},flush=True)
(root/'tuning-resources.json').write_text(json.dumps(resources,indent=2))
