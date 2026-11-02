from common import *
idx=torch.arange(128,device='cuda',dtype=torch.int32)
wa=W.view(torch.uint8)[None].repeat(16,1,1).view(W.dtype);wb=BW.view(torch.uint8)[None].repeat(16,1,1).view(BW.dtype)
views=[(wa[i],wb[i]) for i in range(16)];entries=[]
for split in [1,4]:
 for fused in [False,True]:
  for direct in [False,True]:
   name=f's{split}_fused{fused}_direct{direct}'
   obj=QKVAttentionRegion(W,WS,NW,BW,BS,makecache(),idx,Sink,splits=split,fused=fused,direct=direct)
   obj(X,Cos,Sin);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
    for i in range(128):obj.W,obj.QB=views[i%16];obj(X,Cos,Sin)
   for _ in range(5):g.replay()
   entries.append((name,obj,g))
rows=[]
for rnd in [1,2]:
 samples={n:[] for n,_,_ in entries}
 for i in range(40):
  for n,obj,g in entries if i%2==0 else reversed(entries):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/128)
 row=dict(round=rnd,copies=16,weight_bytes=wa.numel()+wb.numel(),results={n:dict(median_us=statistics.median(samples[n]),samples_us=samples[n],health=obj.check()) for n,obj,_ in entries});rows.append(row)
 (root/'rotating-ab.json').write_text(json.dumps(rows,indent=2));print(rnd,{n:v['median_us'] for n,v in row['results'].items()},flush=True)
