import sys
from pathlib import Path
sys.path.insert(0,'/work/mhc-optimized')
from setup import *
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch as GraphBatch
from native_api import NativeMHC
outdir=Path(__file__).resolve().parent
n=NativeMHC(HAF,HAS,HAB,AN)
r=aiter.mhc_res_shuffle(R.view(1,4,4096)) if False else None
import aiter
r=aiter.mhc_res_shuffle(R.view(1,4,4096));x,p,c=n(r)
def fn():return n.post_pre(x,r,p,c)
for _ in range(3):fn()
g=GraphBatch(fn,128)
for _ in range(5):g.replay()
result=[]
for rnd in [1,2]:
 samples=[]
 for _ in range(40):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/128)
 result.append({'round':rnd,'us':statistics.median(samples),'samples_us':samples})
print(result,flush=True);(outdir/'postpre-timings.json').write_text(json.dumps(result,indent=2))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 fn();torch.cuda.synchronize()
prof.export_chrome_trace(str(outdir/'postpre-trace.json'))
trace=json.loads((outdir/'postpre-trace.json').read_text());kernels=[{'name':e['name'],'us':e['dur'],'args':e.get('args',{})} for e in trace['traceEvents'] if e.get('cat')=='kernel'];(outdir/'postpre-kernels.json').write_text(json.dumps(kernels,indent=2));print(kernels)
