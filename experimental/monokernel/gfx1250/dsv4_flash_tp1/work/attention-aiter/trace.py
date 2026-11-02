from common import *
idx=torch.arange(128,device='cuda',dtype=torch.int32);rows=[]
for fused in [False,True]:
 obj=QKVAttentionRegion(W,WS,NW,BW,BS,makecache(),idx,Sink,fused=fused);obj(X,Cos,Sin);torch.cuda.synchronize()
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):obj(X,Cos,Sin)
 for _ in range(5):cg.replay()
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  cg.replay();torch.cuda.synchronize()
 dest=root/f'chain-trace-{fused}.json';prof.export_chrome_trace(str(dest));trace=json.loads(dest.read_text())
 events=[e for e in trace['traceEvents'] if e.get('cat')=='kernel'];names=[e['name'] for e in events]
 row=dict(fused=fused,kernels=names,count=len(names));rows.append(row);print(row,flush=True)
 assert len(events)==(1 if fused else 2),row
(root/'trace-verification.json').write_text(json.dumps(rows,indent=2))
