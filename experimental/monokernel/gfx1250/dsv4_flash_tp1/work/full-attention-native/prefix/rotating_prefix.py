from setup import *
wa=W.view(torch.uint8)[None].repeat(16,1,1).view(W.dtype);wb=BW.view(torch.uint8)[None].repeat(16,1,1).view(BW.dtype);hw=HAF[None].repeat(16,*([1]*HAF.ndim))
import setup as env
entries=[]
for mode in ['unfused','fused','opt','opt_l2','opt_both']:
 s=State();s.run(R,mode);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for i in range(128):
   s.W,s.QB=wa[i%16],wb[i%16];env.HAF=hw[i%16];s.run(R,mode)
 for _ in range(5):g.replay()
 entries.append((mode,s,g))
rows=[]
for rnd in [1,2]:
 samples={n:[] for n,_,_ in entries}
 for i in range(40):
  for n,s,g in entries if i%2==0 else reversed(entries):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/128)
 row=dict(round=rnd,copies=16,weight_bytes=wa.numel()+wb.numel()+hw.numel()*hw.element_size(),results={n:dict(median_us=statistics.median(samples[n]),samples_us=samples[n],health=s.check()) for n,s,_ in entries});rows.append(row);(root/'prefix-rotating-timings.json').write_text(json.dumps(rows,indent=2));print(rnd,{n:v['median_us'] for n,v in row['results'].items()},flush=True)
