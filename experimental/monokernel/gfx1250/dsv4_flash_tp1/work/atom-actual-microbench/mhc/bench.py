import sys,ast,types
from pathlib import Path
sys.path.insert(0,'/work/mhc-optimized')
from setup import *
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch as GraphBatch
from tdm_tail import candidate
import aiter
outdir=Path(__file__).resolve().parent
source=Path('/app/ATOM/atom/models/deepseek_v4.py').read_text()
tree=ast.parse(source)
cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='hc_pre' for f in c.body))
method=next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='hc_pre')
method_src=ast.unparse(method)
(outdir/'actual_atom_hc_pre.py').write_text(method_src+'\n')
ns={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),'<ATOM hc_pre>','exec'),ns)
obj=types.SimpleNamespace(_mhc_pre=aiter.mhc_pre,enable_hc_fn_pack_bf16=True,norm_eps=1e-6,hc_eps=1e-6,HC_POST_MULT=2.,hc_sinkhorn_iters=20)
packed=aiter.mhc_shuffle_fn(HAF)
Rsh=aiter.mhc_res_shuffle(R.view(1,4,4096))
s=State()
def native(r=Rsh,w=packed):return ns['hc_pre'](obj,r.view(1,4,4096),w,HAS,HAB,AN,1e-6,res_preshuffle=True)
def fused(r=R,w=HAF):return candidate[(256,)](r,w,HAS,HAB,AN,s.mix,s.temp,s.raw,s.x,s.counter,s.health,EARLY=True,**kw)
def fused_adapter():return fused(aiter.mhc_res_unshuffle(Rsh))
errors=[]
for i in range(137):
 r=f2['inputs'][i].cuda().contiguous() if i<9 else torch.randn((4,4096),device='cuda',generator=torch.Generator(device='cuda').manual_seed(171000+i)).bfloat16()
 x,p,c=native(aiter.mhc_res_shuffle(r.view(1,4,4096)));fused(r);torch.cuda.synchronize()
 errors.append({'case':i,'x_relative_rms':err(s.x,x.flatten()),'post_relative_rms':err(s.mix[4:8],p.flatten()),'comb_relative_rms':err(s.mix[8:],c.flatten()),'health':int(s.health.item())})
(outdir/'validation.json').write_text(json.dumps(errors,indent=2))
entries=[]
for mode,fn in [('actual_atom',native),('optimized_fused',fused),('optimized_with_unshuffle',fused_adapter)]:
 for _ in range(3):fn()
 g=GraphBatch(fn,128)
 inner=g.inner
 for _ in range(5):g.replay()
 entries.append((mode,g,inner))
results=[]
for rnd in [1,2]:
 samples={n:[] for n,g,inner in entries}
 for j in range(40):
  for n,g,inner in entries if j%2==0 else reversed(entries):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/128)
 row={'round':rnd,'us':{n:statistics.median(v) for n,v in samples.items()},'samples_us':samples};results.append(row);print(row['us'],flush=True)
(outdir/'timings.json').write_text(json.dumps(results,indent=2))
for n,fn in [('actual_atom',native),('optimized_fused',fused),('optimized_with_unshuffle',fused_adapter)]:
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
  fn();torch.cuda.synchronize()
 prof.export_chrome_trace(str(outdir/(n+'-trace.json')))
 trace=json.loads((outdir/(n+'-trace.json')).read_text())
 kernels=[{'name':e['name'],'us':e['dur'],'args':e.get('args',{})} for e in trace['traceEvents'] if e.get('cat')=='kernel']
 (outdir/(n+'-kernels.json')).write_text(json.dumps(kernels,indent=2))
 print(n,kernels,flush=True)
print('max_errors',{k:max(e[k] for e in errors) for k in errors[0] if k!='case'},flush=True)
(outdir/'metadata.json').write_text(json.dumps({'atom_class':cls.name,'source':'/app/ATOM/atom/models/deepseek_v4.py','method_line':method.lineno,'packed_dtype':str(packed.dtype),'input_shape':[1,4,4096],'weight_shape':list(HAF.shape),'norm_weight_shape':[4096],'norm_fused':True,'round_to_report':2,'graph_repetitions':128,'allocation_policy':'single captured call, kernelnodes flat-cloned128 times, stable addresses','residual_preshuffle':True,'native_launches':3,'fused_core_launches':1,'fused_matched_input_launches':2,'samples_per_round':40,'checkpoint':'/data/DeepSeek-V4-Flash','scope':'layer0 attention mHC pre + norm; no preceding mHC post; independent of context length'},indent=2))
