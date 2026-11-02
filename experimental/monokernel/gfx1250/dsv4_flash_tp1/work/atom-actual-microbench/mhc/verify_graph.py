from pathlib import Path
exec(compile(Path(__file__).with_name('bench.py').read_text().split('errors=[]')[0],str(Path(__file__).with_name('bench.py')),'exec'))
from collections import Counter
checks=[]
for name,fn,nkern in [('actual_atom',native,3),('optimized_fused',fused,1),('optimized_with_unshuffle',fused_adapter,2)]:
 for _ in range(3):fn()
 g=GraphBatch(fn,8);g.replay();torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  g.replay();torch.cuda.synchronize()
 path=outdir/(name+'-flatgraph-trace.json');prof.export_chrome_trace(str(path))
 names=Counter(e['name'] for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel')
 assert sum(names.values())==nkern*8,(name,names)
 checks.append({'name':name,'kernel_count':sum(names.values()),'expected':nkern*8,'names':dict(names)})
(outdir/'flatgraph-verification.json').write_text(json.dumps(checks,indent=2));print(checks)
