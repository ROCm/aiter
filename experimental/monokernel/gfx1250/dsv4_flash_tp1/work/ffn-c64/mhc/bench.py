import sys,json,statistics,importlib.util
from pathlib import Path
import torch,aiter
from safetensors import safe_open
from aiter.ops.quant import per_group_quant_hip
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch
root=Path(__file__).resolve().parent
model=Path('/data/DeepSeek-V4-Flash');index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
if not (root/'fixture.pt').exists():
 data={}
 for name,key in [('fn','hc_ffn_fn'),('scale','hc_ffn_scale'),('base','hc_ffn_base'),('norm','ffn_norm.weight')]:
  full='layers.0.'+key
  with safe_open(str(model/index[full]),framework='pt',device='cpu') as sf:data[name]=sf.get_tensor(full)
 torch.save(data,root/'fixture.pt')
f=torch.load(root/'fixture.pt',weights_only=True);fn=aiter.mhc_shuffle_fn(f['fn'].cuda().contiguous());scale=f['scale'].cuda();base=f['base'].cuda();norm=f['norm'].cuda()
gen=torch.Generator(device='cuda').manual_seed(6401024)
x=torch.randn((64,4096),device='cuda',generator=gen).bfloat16();rp=torch.randn((64,4,4096),device='cuda',generator=gen).bfloat16();r=aiter.mhc_res_shuffle(rp)
post=torch.sigmoid(torch.randn((64,4,1),device='cuda',generator=gen))*2
comb=torch.softmax(torch.randn((64,4,4),device='cuda',generator=gen),dim=-1)
spec=importlib.util.spec_from_file_location('old_hybrid','/work/flash/candidate/dsv4_hybrid.py');hy=importlib.util.module_from_spec(spec);spec.loader.exec_module(hy)
def native():return aiter.mhc_fused_post_pre(x,r,post,comb,fn,scale,base,1e-6,1e-6,1e-6,2.,20,norm,1e-6,w_preshuffle_bf16=True,res_preshuffle=True)
def quant(y):return per_group_quant_hip(y,quant_dtype=torch.float8_e4m3fn,group_size=128,transpose_scale=True,scale_type=torch.float8_e8m0fnu)
def native_quant():
 out=native();return (*out,*quant(out[2]))
def hybrid():return hy.dsv4_hybrid_hc(x,r,post,comb,fn,scale,base,norm,res_preshuffle=True)
# This pre-only baseline starts AFTER post. Prepare next residual once outside timing.
post_res=native()[3]
def pre_only():return aiter.mhc_pre(post_res,fn,scale,base,1e-6,1e-6,1e-6,2.,20,norm,1e-6,w_preshuffle_bf16=1,res_preshuffle=True)
rows=[]
for case in range(16):
 x.copy_(torch.randn(x.shape,device='cuda',generator=gen).bfloat16());rp.copy_(torch.randn(rp.shape,device='cuda',generator=gen).bfloat16());r.copy_(aiter.mhc_res_shuffle(rp))
 a=native_quant();b=hybrid();torch.cuda.synchronize()
 errs=[]
 for aa,bb in zip(a,b):
  if aa.dtype==torch.float8_e8m0fnu:errs.append({'bitwise':torch.equal(aa.view(torch.uint8),bb.view(torch.uint8)),'rms':None})
  else:errs.append({'bitwise':torch.equal(aa,bb),'rms':float(((aa.float()-bb.float()).square().mean()/aa.float().square().mean().clamp_min(1e-30)).sqrt())})
 rows.append({'case':case,'outputs':errs})
(root/'validation.json').write_text(json.dumps(rows,indent=2));print('validation',rows[0],flush=True)
funcs={'actual_post_pre_norm':native,'actual_post_pre_norm_quant':native_quant,'hybrid_post_pre_norm_quant':hybrid,'actual_ffn_pre_norm_only':pre_only}
entries=[]
for name,call in funcs.items():
 for _ in range(3):call()
 gr=FlatGraphBatch(call,128)
 for _ in range(5):gr.replay()
 entries.append((name,gr));print('ready',name,gr.kernel_count,flush=True)
results=[]
for rnd in [1,2]:
 samples={name:[] for name,gr in entries}
 for i in range(40):
  for name,gr in entries if i%2 else reversed(entries):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();gr.replay();b.record();b.synchronize();samples[name].append(a.elapsed_time(b)*1000/128)
 row={'round':rnd,'us':{name:statistics.median(vals) for name,vals in samples.items()},'samples_us':samples};results.append(row);print(row['us'],flush=True)
(root/'timings.json').write_text(json.dumps(results,indent=2))
for name,call in funcs.items():
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  call();torch.cuda.synchronize()
 path=root/(name+'-trace.json');prof.export_chrome_trace(str(path))
 kernels=[{'name':e['name'],'us':e['dur'],'args':e.get('args',{})} for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel'];(root/(name+'-kernels.json')).write_text(json.dumps(kernels,indent=2));print(name,kernels,flush=True)
