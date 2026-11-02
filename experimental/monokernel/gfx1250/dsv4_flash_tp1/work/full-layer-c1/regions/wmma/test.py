exec(compile((__import__('pathlib').Path(__file__).parent/'setup.py').read_text(),'setup.py','exec'))
from split_control import split_stage
from region_kernel import region_kernel
import importlib.util,statistics,re
spec=importlib.util.spec_from_file_location('vector_region',root.parent/'region_kernel.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
kw=dict(GRID=args.grid,num_warps=4,enable_fp_fusion=False)
def split():
 for i in range(5,11):split_stage[(args.grid,)](*values,STAGE=i,**kw)
def vector():return mod.region_kernel[(args.grid,)](*values,REGION=1,**kw)
extra=dict(XP=torch.empty(4096,device='cuda',dtype=torch.float8_e4m3fn),XS=torch.empty(32,device='cuda',dtype=torch.uint8),LP=torch.empty(1024,device='cuda',dtype=torch.float8_e4m3fn),LS=torch.empty(8,device='cuda',dtype=torch.uint8))
extra.update(QPART=torch.empty(8192,device='cuda',dtype=torch.float32),KPART=torch.empty(4096,device='cuda',dtype=torch.float32))
def wmma():return region_kernel[(args.grid,)](*values,REGION=1,**kw,**extra)
for i in range(5):split_stage[(args.grid,)](*values,STAGE=i,**kw)
split();torch.cuda.synchronize()
expected={k:d[k].clone() for k in ['QL','KV','QLQ','Q','QRot','KVRot']}
c=wmma();torch.cuda.synchronize()
(root/'qkv.amdgcn').write_text(c.asm['amdgcn'])
assert 'v_wmma' in c.asm['amdgcn']
checks={}
for k,v in expected.items():
 diff=d[k].float()-v.float();rr=(diff.square().mean()/v.float().square().mean()).sqrt().item()
 checks[k]=dict(relative_rms=rr,max_abs=diff.abs().max().item(),exact=torch.equal(d[k],v))
print('correctness',checks,flush=True)
(root/'correctness.json').write_text(json.dumps(checks,indent=2))
assert all(x['relative_rms']<.005 for x in checks.values()),checks
assert d['Health'].item()==0
resources=dict(n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared,sgpr_spills=re.findall(r'\.sgpr_spill_count:\s*(\d+)',c.asm['amdgcn']),vgpr_spills=re.findall(r'\.vgpr_spill_count:\s*(\d+)',c.asm['amdgcn']))
print('resources',resources,flush=True);(root/'resources.json').write_text(json.dumps(resources,indent=2))
def measure(fn,name):
 fn();torch.cuda.synchronize();cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):fn()
 for _ in range(10):cg.replay()
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  for _ in range(30):cg.replay()
  torch.cuda.synchronize()
 path=root/(name+'.json');prof.export_chrome_trace(str(path))
 es=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel']
 n=len(es)//30;assert len(es)==30*n and n>0
 samples=[sum(e['dur'] for e in es[i:i+n]) for i in range(0,len(es),n)]
 return dict(median_us=statistics.median(samples),samples_us=samples,kernels=n)
rows=[]
for run in [1,2]:
 row=dict(run=run,split=measure(split,f'split-{run}'),vector=measure(vector,f'vector-{run}'),wmma=measure(wmma,f'wmma-{run}'))
 rows.append(row);print('timing',row,flush=True)
(root/'timings.json').write_text(json.dumps(rows,indent=2))
print('DONE',flush=True)
from wmma_ops import projection_probe
# Check each raw WMMA projection against independent FP32 torch matmul on the
# exact packed operands used by that projection (separate from propagation).
projection_checks={}
for name,x,xs,w,ws,n,k in [('QA',extra['XP'],extra['XS'],d['QA'],d['QAS'],1024,4096),('KV',extra['XP'],extra['XS'],d['KVW'],d['KVS'],512,4096),('QB',extra['LP'],extra['LS'],d['QB'],d['QBS'],32768,1024)]:
 out=torch.empty(n,device='cuda',dtype=torch.bfloat16)
 projection_probe[(256,)](x,xs,w,ws,out,n,k,256,num_warps=4,enable_fp_fusion=False)
 wf=w.float()*torch.pow(2.,ws.float()-127).repeat_interleave(128,0).repeat_interleave(128,1)
 xf=x.float()*torch.pow(2.,xs.float()-127).repeat_interleave(128)
 ref=torch.mv(wf,xf).bfloat16()
 delta=out.float()-ref.float()
 rr=(delta.square().mean()/ref.float().square().mean()).sqrt().item()
 projection_checks[name]=dict(relative_rms=rr,max_abs=delta.abs().max().item(),exact=torch.equal(out,ref))
 assert rr<.005,projection_checks
 del wf,xf,ref,out
(root/'projection-reference.json').write_text(json.dumps(projection_checks,indent=2))
print('independent projections',projection_checks,flush=True)
# New residual inputs derived from actual embedding rows, not just one fixture.
saved_r=d['R'].clone();cases=[]
from reference import fp8, norm as refnorm, quant as refquant, rope as refrope
rw={name:fp8(d[name],d[sc]) for name,sc in [('QA','QAS'),('KVW','KVS'),('QB','QBS')]}
def independent():
 ql=refnorm((rw['QA']@d['XQ']).bfloat16(),d['QN'],False)
 kv=refnorm((rw['KVW']@d['XQ']).bfloat16(),d['KN'],False)
 qlq=refquant(ql,128)
 q=refnorm((rw['QB']@qlq).bfloat16().reshape(64,512),rounded=False).flatten()
 return dict(QL=ql,KV=kv,QLQ=qlq,Q=q,QRot=refrope(q,d['Cos'],d['Sin']).flatten(),KVRot=refrope(kv,d['Cos'],d['Sin']).flatten())
def compare(actual,ref):
 return {k:((actual[k].float()-v.float()).square().mean()/v.float().square().mean()).sqrt().item() for k,v in ref.items()}

for tok in [0,1,7,43,100,1024,8192,32000]:
 with safe_open(str(p/index['embed.weight']),framework='pt',device='cpu') as f:r=f.get_slice('embed.weight')[tok:tok+1].cuda()
 d['R'].copy_(r.repeat(4,1))
 for st in range(5):split_stage[(args.grid,)](*values,STAGE=st,**kw)
 split();expected_case={k:d[k].clone() for k in expected}
 wmma();torch.cuda.synchronize()
 metrics={k:((d[k].float()-v.float()).square().mean()/v.float().square().mean()).sqrt().item() for k,v in expected_case.items()}
 assert all(torch.isfinite(d[k]).all().item() for k in expected)
 ref=independent()
 assert d['Health'].item()==0
 cases.append(dict(token=tok,relative_rms=metrics,passes_original_threshold=max(metrics.values())<.005,wmma_vs_reference=compare(d,ref),vector_vs_reference=compare(expected_case,ref)))
d['R'].copy_(saved_r)
for st in range(5):split_stage[(args.grid,)](*values,STAGE=st,**kw)
# Grid counter wrap and changing-input graph replay validation.
for initial in [2147483648-256,-256]:
 d['Counter'].fill_(initial);wmma();torch.cuda.synchronize();assert d['Health'].item()==0
cg=torch.cuda.CUDAGraph()
with torch.cuda.graph(cg):
 d['XQ'].neg_();wmma()
for _ in range(1000):cg.replay()
torch.cuda.synchronize();assert d['Health'].item()==0
actual={k:d[k].clone() for k in expected};split();torch.cuda.synchronize()
metrics={k:((actual[k].float()-d[k].float()).square().mean()/d[k].float().square().mean()).sqrt().item() for k in expected}
assert max(metrics.values())<.005,metrics
(root/'extended-validation.json').write_text(json.dumps(dict(embedding_cases=cases,graph_replays=1000,graph_relative_rms=metrics,counter_wrap=True,health=d['Health'].item()),indent=2))
print('Extended numerical comparisons',cases,flush=True)
print('Graph replay and wrap passed. Numerical threshold passed for all embeddings:',all(c['passes_original_threshold'] for c in cases),flush=True)
assert all(c['passes_original_threshold'] for c in cases), 'Original numerical gate failed'
assert max(max(c['wmma_vs_reference'].values()) for c in cases)<1e-4, 'Independent reference gate failed'
print('ACCEPTED: isolated region numerical and replay gates passed; not serving validation',flush=True)
