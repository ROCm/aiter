from native import *
import statistics,time,collections
sys.path.insert(0,'/work/mhc-optimized')
from qkv_region import QKVRegion
from split_kernels import cache_store
n=NativeQKV();f=n.f
r=QKVRegion(f['W_bytes'].cuda().view(torch.float8_e4m3fn),f['WS'].cuda(),f['NW'].cuda(),f['QB_bytes'].cuda().view(torch.float8_e4m3fn),f['QBS'].cuda())
c=n.cos[1024].flatten().float().contiguous();s=n.sin[1024].flatten().float().contiguous()
def fused():return r(n.x,c,s)
def fused_scatter():
 q,k=fused();cache_store[(1,)](k,n.cache,SLOT=0,num_warps=4,enable_fp_fusion=False);return q,k
def error(a,b):return ((a.float()-b.float()).square().mean()/b.float().square().mean().clamp_min(1e-30)).sqrt().item()
funcs={'native_with_scatter':n,'native_without_scatter':lambda:n(scatter=False),'fused':fused,'fused_plus_scatter':fused_scatter}
for fn in funcs.values():
 for i in range(3):fn()
torch.cuda.synchronize()
q,k=n();q=q.clone();k=k.clone();fq,fk=fused();torch.cuda.synchronize()
validation={'q_relative_rms':error(fq,q),'kv_relative_rms':error(fk,k),'q_max_abs':(fq.float()-q.float()).abs().max().item(),'kv_max_abs':(fk.float()-k.float()).abs().max().item(),'health':r.health.item()}
json.dump(validation,open('validation.json','w'),indent=2);print('validation',validation,flush=True)
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import GraphBatch
graphs={name:GraphBatch(fn,128) for name,fn in funcs.items()}
for g in graphs.values():
 for _ in range(5):g.replay()
torch.cuda.synchronize();rounds=[]
for rnd in range(2):
 raw={k:[] for k in graphs}
 for i in range(40):
  keys=list(graphs);keys=keys if i%2==0 else keys[::-1]
  for k in keys:
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();graphs[k].replay();b.record();b.synchronize();raw[k].append(a.elapsed_time(b)*1000/128)
 result={k:statistics.median(v) for k,v in raw.items()};rounds.append({'median_us':result,'samples_us':raw});print('round',rnd+1,result,flush=True)
json.dump({'rounds':rounds,'metadata':n.metadata(),'validation':validation,'method':{'calls_per_graph':128,'allocation':'capture one call, outer128 dependent HIP childgraph nodes same buffers','warmups':5,'samples':40,'report_round':2}},open('fixed-results.json','w'),indent=2)
checks={}
for name,g in graphs.items():
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:g.replay();torch.cuda.synchronize()
 p.export_chrome_trace('fixed-'+name+'-trace.json')
 ev=json.load(open('fixed-'+name+'-trace.json'))['traceEvents'];ks=[e for e in ev if e.get('cat')=='kernel'];counts=collections.Counter(e['name'] for e in ks)
 checks[name]={'kernel_count':len(ks),'counts':dict(counts),'kernel_durations_us_sum_per_invocation':sum(e['dur'] for e in ks)/128}
json.dump(checks,open('fixed-profile-checks.json','w'),indent=2);print('checks',checks,flush=True)
