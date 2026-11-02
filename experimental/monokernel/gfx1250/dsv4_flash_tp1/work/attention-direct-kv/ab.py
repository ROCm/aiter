from common import *
entries=[];resources={}
for t in [128,1024]:
 for s in [1,4]:
  for kind in ['attention','two','fused']:
   for direct in [False,True]:
    name=f't{t}_s{s}_{kind}_direct{direct}'
    cache=makecache(t);idx=torch.arange(t,device='cuda',dtype=torch.int32)
    if kind=='attention':
     obj=AttentionRegion(cache,idx,Sink,splits=s,direct=direct);args=(Q,KV)
    else:
     obj=QKVAttentionRegion(W,WS,NW,BW,BS,cache,idx,Sink,splits=s,fused=kind=='fused',direct=direct);args=(X,Cos,Sin)
    obj(*args);torch.cuda.synchronize()
    k=obj.obj if kind!='two' else obj.attn.obj
    resources[name]=dict(vgpr=k.n_regs,lds=k.metadata.shared,spills=k.n_spills)
    (root/f'{name}.amdgcn').write_text(k.asm['amdgcn'])
    cg=torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
     for _ in range(128):obj(*args)
    for _ in range(5):cg.replay()
    entries.append((name,obj,cg))
    print('ready',name,flush=True)
(root/'ab-resources.json').write_text(json.dumps(resources,indent=2))
results=[]
for rnd in [1,2]:
 samples={n:[] for n,_,_ in entries}
 for i in range(40):
  for n,o,g in entries if i%2==0 else reversed(entries):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record();g.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/128)
 row=dict(round=rnd,results={n:dict(median_us=statistics.median(samples[n]),samples_us=samples[n],health=o.check()) for n,o,_ in entries})
 results.append(row);(root/'ab-timings.json').write_text(json.dumps(results,indent=2));print(rnd,{n:v['median_us'] for n,v in row['results'].items()},flush=True)
