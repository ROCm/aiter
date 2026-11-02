from setup import *
rows=[]
for mode,expected in [('unfused',17),('three_regions',3),('fused',1)]:
 s=State();s.run(R,mode);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):s.run(R,mode)
 for _ in range(5):g.replay()
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  g.replay();torch.cuda.synchronize()
 dest=root/f'{mode}-trace.json';prof.export_chrome_trace(str(dest));tr=json.loads(dest.read_text());events=[e for e in tr['traceEvents'] if e.get('cat')=='kernel']
 row=dict(mode=mode,count=len(events),names=[e['name'] for e in events]);print(row,flush=True);assert len(events)==expected
 rows.append(row)
(root/'trace-verification.json').write_text(json.dumps(rows,indent=2))
