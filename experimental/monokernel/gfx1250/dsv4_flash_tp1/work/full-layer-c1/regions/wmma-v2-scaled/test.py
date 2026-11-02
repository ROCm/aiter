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
