import os,sys,json,statistics
from pathlib import Path
os.environ['ATOM_USE_TRITON_GEMM']='1'
os.environ['ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE']='1'
os.environ['ATOM_USE_TRITON_MOE']='1'
os.environ['AITER_TRITON_USE_HERD']='1'
sys.path[:0]=['/work/c8-ablation','/work/batch-full','/work/modelwide','/work/concurrency-rollout','/work/full-attention-native/prefix','/work/ffn-c1/moe','/work/atom-actual-microbench/attention','/app/aiter','/app/ATOM']
import torch,aiter
from graphbatch import FlatGraphBatch
ROOT=Path('/work/c8-ablation')
M=8
kw=dict(num_warps=4,enable_fp_fusion=False)
def err(a,b):
 a=a.float().flatten();b=b.float().flatten()
 return dict(finite=bool(torch.isfinite(a).all()),relative_rms=float(((a-b).square().mean()/b.square().mean().clamp_min(1e-30)).sqrt()),max_abs=float((a-b).abs().max()))
def bench(name,fs,validation=None):
 gs={}
 for k,fn in fs.items():
  for _ in range(3):fn()
  torch.cuda.synchronize();gs[k]=FlatGraphBatch(fn,32)
 rounds=[]
 for rnd in [1,2]:
  for g in gs.values():
   for _ in range(5):g.replay()
  samples={k:[] for k in gs}
  for it in range(40):
   for k in list(gs)[::1 if it%2 else -1]:
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();gs[k].replay();b.record();b.synchronize();samples[k].append(a.elapsed_time(b)*1000/32)
  rounds.append(dict(round=rnd,median_us={k:statistics.median(v) for k,v in samples.items()},samples_us=samples))
 result=dict(name=name,M=8,method='flat HIP graph 32 repetitions; 5 warmups; 40 alternating paired samples; second round reported',rounds=rounds,validation=validation,kernel_count={k:g.kernel_count//32 for k,g in gs.items()})
 (ROOT/(name+'.json')).write_text(json.dumps(result,indent=2));print(name,rounds[-1]['median_us'],validation,flush=True)
 for k,fn in fs.items():
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
   fn();torch.cuda.synchronize()
  path=ROOT/(name+'-'+k+'-trace.json');p.export_chrome_trace(str(path))
  ev=json.loads(path.read_text())['traceEvents'];result.setdefault('kernels',{})[k]=[dict(name=x['name'],us=x['dur']) for x in ev if x.get('cat')=='kernel']
 (ROOT/(name+'.json')).write_text(json.dumps(result,indent=2))
 return result
