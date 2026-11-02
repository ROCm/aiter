import sys
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch as GraphBatch
import sys,json,statistics
from pathlib import Path
sys.path[:0]=['/work/atom-actual-microbench/qkv','/work/atom-actual-microbench/mhc','/work/mhc-optimized','/app/aiter','/app/ATOM']
from native import NativeQKV
from native_api import NativeMHC
sys.path.insert(0,'/work/mhc-optimized')
from setup import State,R,HAF,HAS,HAB,AN,err
import aiter,torch
from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
outdir=Path(__file__).resolve().parent
m=NativeMHC(HAF,HAS,HAB,AN)
s=State(128);r=aiter.mhc_res_shuffle(R.view(1,4,4096))
n=NativeQKV(cache=s.attn.cache.clone(),dest=127)
import setup as candidate_setup
candidate_setup.Cos=n.cos[1024].flatten().float().contiguous();candidate_setup.Sin=n.sin[1024].flatten().float().contiguous()
idx=torch.zeros(128*128,device='cuda',dtype=torch.int32);idx[:128]=torch.arange(128,device='cuda',dtype=torch.int32)
ptr=torch.tensor([0,128],device='cuda',dtype=torch.int32)
def actual():
 x,post,comb=m(r);q,kv=n(x)
 return sparse_attn_v4_paged_decode(q,n.cache,idx,ptr,s.attn.sink,512**-.5)
def candidate():return s.run(aiter.mhc_res_unshuffle(r),'opt_both')
def candidate_core():return s.run(R,'opt_both')
functions={'actual_atom':actual,'fused_with_input_adapter':candidate,'fused_core_plain_input':candidate_core}
graphs={}
ref=actual().clone();out=candidate().clone();torch.cuda.synchronize()
validation={'attention_output_rel_rms':err(out,ref),'health':s.check(),'threshold':.001,'note':'Core uses plain residual; adapter row starts with actual ATOM shuffled residual. Native cached indices capacity16384, true128. Different native arithmetic may fail gate; timing is not correctness acceptance.'}
for name,fn in functions.items():
 for _ in range(4):fn()
 torch.cuda.synchronize();g=GraphBatch(fn)
 graphs[name]=g
 print('ready',name,flush=True)
rounds=[]
for rnd in [1,2]:
 for g in graphs.values():
  for _ in range(5):g.replay()
 samples={k:[] for k in graphs}
 for i in range(40):
  for k in list(graphs)[::1 if i%2==0 else -1]:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   a.record();graphs[k].replay();b.record();b.synchronize();samples[k].append(a.elapsed_time(b)*1000/128)
 rounds.append({'round':rnd,'medians_us':{k:statistics.median(v) for k,v in samples.items()},'samples_us':samples})
 print(rounds[-1]['medians_us'],flush=True)
 (outdir/'timings.json').write_text(json.dumps(rounds,indent=2))
validation['health_after_timing']=s.check();(outdir/'validation.json').write_text(json.dumps(validation,indent=2))
for name,fn in functions.items():
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
  fn();torch.cuda.synchronize()
 p.export_chrome_trace(str(outdir/f'{name}-trace.json'))
 e=json.loads((outdir/f'{name}-trace.json').read_text())['traceEvents']
 (outdir/f'{name}-kernels.json').write_text(json.dumps([{'name':x['name'],'us':x['dur'],'args':x['args']} for x in e if x.get('cat')=='kernel'],indent=2))
print(validation,flush=True)
counts={}
for name,g in graphs.items():
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
  g.replay();torch.cuda.synchronize()
 p.export_chrome_trace(str(outdir/f'{name}-batch-trace.json'))
 e=json.loads((outdir/f'{name}-batch-trace.json').read_text())['traceEvents'];ks=[x for x in e if x.get('cat')=='kernel'];counts[name]={'kernel_count':len(ks),'kernel_total_us':sum(x['dur'] for x in ks),'kernel_names':sorted(set(x['name'] for x in ks))}
(outdir/'batch-verification.json').write_text(json.dumps(counts,indent=2));print(counts,flush=True)

(outdir/'graph-node-counts.json').write_text(json.dumps({k:{'kernel_count':g.kernel_count,'original_types':g.node_types} for k,g in graphs.items()},indent=2))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
 for k,g in graphs.items():
  with torch.profiler.record_function(k):g.replay();torch.cuda.synchronize()
p.export_chrome_trace(str(outdir/'all-batch-trace.json'))
