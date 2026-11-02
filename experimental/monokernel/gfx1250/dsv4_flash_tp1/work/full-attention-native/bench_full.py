import os
from full_setup import *
import statistics
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch
outdir=Path(os.environ['OUTDIR']) if 'OUTDIR' in os.environ else root
outdir.mkdir(exist_ok=True)
f=FullState();r=aiter.mhc_res_shuffle(ps.R.view(1,4,4096))
funcs={'actual_atom_extracted_boundary':lambda:f.native(r),'fused_single_launch':lambda:f.fused(r),'staged_prototype_control':lambda:f.staged(r)}
graphs={}
with torch.inference_mode():
 for name,fn in funcs.items():
  for _ in range(3):fn()
  torch.cuda.synchronize();graphs[name]=FlatGraphBatch(fn,128);print('ready',name,graphs[name].kernel_count,flush=True)
 rounds=[]
 for rnd in [1,2]:
  for g in graphs.values():
   for _ in range(5):g.replay()
  samples={k:[] for k in graphs}
  for i in range(40):
   for name in list(graphs)[::1 if i%2==0 else -1]:
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();graphs[name].replay();b.record();b.synchronize();samples[name].append(a.elapsed_time(b)*1000/128)
  item={'round':rnd,'median_us':{k:statistics.median(v) for k,v in samples.items()},'samples_us':samples};rounds.append(item);(outdir/'timings.json').write_text(json.dumps(rounds,indent=2));print(item['median_us'],flush=True)
 (outdir/'graph-counts.json').write_text(json.dumps({k:{'kernel_nodes':g.kernel_count,'inner_node_types':g.node_types} for k,g in graphs.items()},indent=2))
 checks={}
 for name,g in graphs.items():
  expected=funcs[name]().clone();torch.cuda.synchronize();g.out.fill_(float('nan'));g.replay();torch.cuda.synchronize()
  checks[name]={'poison_replaced':bool(torch.isfinite(g.out).all()),'exact_eager':torch.equal(g.out.flatten(),expected.flatten()),'health':f.check()}
 (outdir/'graph-validation.json').write_text(json.dumps(checks,indent=2))
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
  for name,fn in funcs.items():
   with torch.profiler.record_function(name):fn();torch.cuda.synchronize()
 p.export_chrome_trace(str(outdir/'trace.json'))
 e=json.loads((outdir/'trace.json').read_text())['traceEvents'];(outdir/'kernels.json').write_text(json.dumps([{'name':x['name'],'us':x['dur']} for x in e if x.get('cat')=='kernel'],indent=2))
 print(checks,flush=True)
