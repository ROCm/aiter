from layer_setup import *
import statistics
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch
f=LayerState()
def err(a,b):return float(((a.float()-b.float()).square().mean()/b.float().square().mean().clamp_min(1e-30)).sqrt())
checks=[]
inputs=list(fa.ps.f2['inputs'])
gen=torch.Generator().manual_seed(20261020)
inputs += [torch.randn(4,4096,generator=gen).bfloat16()*.05 for _ in range(8)]
for pos in [1024,1536,2047]:
 f.a.setpos(pos)
 for case,r in enumerate(inputs):
  f.r.copy_(aiter.mhc_res_shuffle(r.cuda().view(1,4,4096)));ref=f.native().clone();out=f.fused().clone();split=f.staged().clone();torch.cuda.synchronize()
  checks.append({'position':pos,'case':case,'native_rms':err(out,ref),'staged_exact':torch.equal(out,split),'finite':bool(torch.isfinite(out).all()),'health':f.health()})
(root/'validation.json').write_text(json.dumps(checks,indent=2));print('validation',len(checks),'max',max(x['native_rms'] for x in checks),'staged',all(x['staged_exact'] for x in checks),flush=True)
f.a.setpos(1024);f.r.copy_(aiter.mhc_res_shuffle(fa.ps.R.view(1,4,4096)))
fs={'native':f.native,'two_fused_regions':f.staged,'single_full_layer':f.fused};gs={}
for n,fn in fs.items():
 for _ in range(3):fn()
 torch.cuda.synchronize();gs[n]=FlatGraphBatch(fn,16)
rounds=[]
for rnd in [1,2]:
 for gr in gs.values():
  for _ in range(5):gr.replay()
 raw={k:[] for k in fs}
 for i in range(40):
  for n in list(fs)[::1 if i%2 else -1]:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();gs[n].replay();b.record();b.synchronize();raw[n].append(a.elapsed_time(b)*1000/16)
 rounds.append({'round':rnd,'median_us':{n:statistics.median(v) for n,v in raw.items()},'samples_us':raw});print(rounds[-1]['median_us'],flush=True)
(root/'timings.json').write_text(json.dumps(rounds,indent=2));(root/'resources.json').write_text(json.dumps({'vgpr':f.obj.n_regs,'lds':f.obj.metadata.shared,'spills':f.obj.n_spills,'graph_nodes':{n:g.kernel_count for n,g in gs.items()},'health':f.health()},indent=2))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
 for fn in fs.values():fn()
 torch.cuda.synchronize()
p.export_chrome_trace(str(root/'trace.json'))
