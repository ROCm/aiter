exec(compile((__import__('pathlib').Path(__file__).parent/'setup.py').read_text(),'setup.py','exec'))
from split_control import split_stage
from region_kernel import region_kernel
from full_kernel import all_wmma_layer
import statistics,re,importlib.util
kw=dict(GRID=args.grid,num_warps=4,enable_fp_fusion=False)
extra=dict(XP=torch.empty(4096,device='cuda',dtype=torch.float8_e4m3fn),XS=torch.empty(32,device='cuda',dtype=torch.uint8),LP=torch.empty(1024,device='cuda',dtype=torch.float8_e4m3fn),LS=torch.empty(8,device='cuda',dtype=torch.uint8),QPART=torch.empty(8192,device='cuda',dtype=torch.float32),KPART=torch.empty(4096,device='cuda',dtype=torch.float32),MHC_TEMP=torch.empty(128*24,device='cuda'))
regions=[(0,4),(5,10),(11,12),(13,17),(18,22),(23,30),(31,32)]
watch=[*sizes,'Out','Cache']
def split(a=0,b=32):
 for i in range(a,b+1):split_stage[(args.grid,)](*values,STAGE=i,**kw)
def region(i):return region_kernel[(args.grid,)](*values,REGION=i,**kw,**extra)
def full():return all_wmma_layer[(args.grid,)](*values,**kw,**extra)
def snapshot():return {k:d[k].clone() for k in watch}
def restore(s):
 for k,v in s.items():d[k].copy_(v)
def compare(s):
 result={}
 for k,v in s.items():
  diff=d[k].float()-v.float()
  result[k]=dict(relative_rms=(diff.square().mean()/v.float().square().mean().clamp_min(1e-30)).sqrt().item(),max_abs=diff.abs().max().item(),exact=torch.equal(d[k],v),finite=torch.isfinite(d[k]).all().item())
 return result
def audit(c,name):
 isa=c.asm['amdgcn'];(root/(name+'.amdgcn')).write_text(isa)
 return dict(n_regs=c.n_regs,shared=c.metadata.shared,sgpr_spills=re.findall(r'\.sgpr_spill_count:\s*(\d+)',isa),vgpr_spills=re.findall(r'\.vgpr_spill_count:\s*(\d+)',isa),scratch=re.findall(r'ScratchSize:\s*(\d+)',isa),wmma=sorted(set(re.findall(r'v_wmma\w+',isa))))
split();torch.cuda.synchronize();baseline=snapshot();fixtures=[];resources={};checks={}
for i,(a,b) in enumerate(regions):
 split(0,a-1);before=snapshot();fixtures.append(before)
 split(a,b);torch.cuda.synchronize();expected=snapshot()
 restore(before);print('compile region',i,flush=True);c=region(i);torch.cuda.synchronize()
 resources[str(i)]=audit(c,'region-'+str(i));checks[str(i)]=compare(expected)
 print('region check',i,'max rms',max(x['relative_rms'] for x in checks[str(i)].values()),resources[str(i)],flush=True)
 assert d['Health'].item()==0
 assert all(x['finite'] for x in checks[str(i)].values())
 (root/'region-checks.json').write_text(json.dumps(checks,indent=2));(root/'resources.json').write_text(json.dumps(resources,indent=2))
print('compile full',flush=True);c=all_wmma_layer.warmup(*values,grid=(args.grid,),**kw,**extra);c._init_handles();resources['full']=audit(c,'full');(root/'resources.json').write_text(json.dumps(resources,indent=2));print('full resources',resources['full'],flush=True)
assert c.metadata.shared<=196608
full();torch.cuda.synchronize();assert d['Health'].item()==0
(root/'full-vs-vector.json').write_text(json.dumps(compare(baseline),indent=2))
from reference import check
check(d,root)
print('FULL REFERENCE PASSED',flush=True)
import ctypes
hip=ctypes.CDLL('libamdhip64.so');occ=ctypes.c_int()
f=hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
f.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t];f.restype=ctypes.c_int
assert f(ctypes.byref(occ),ctypes.c_void_p(c.function),128,c.metadata.shared)==0
assert occ.value>=1
resources['full']['blocks_per_cu']=occ.value
(root/'resources.json').write_text(json.dumps(resources,indent=2))
# Confirm full-layer and seven-region executions have identical arithmetic.
full();expected_full=snapshot()
for i in range(7):region(i)
torch.cuda.synchronize();sequence=compare(expected_full)
(root/'full-vs-regions.json').write_text(json.dumps(sequence,indent=2))
assert all(x['exact'] for x in sequence.values()),sequence
# Multiple real embeddings and corresponding hash routing IDs, independent ref.
saved_r=d['R'].clone();cases=[]
for token_id in [0,1,7,43]:
 with safe_open(str(p/index['embed.weight']),framework='pt',device='cpu') as sf:emb=sf.get_slice('embed.weight')[token_id:token_id+1].cuda()
 d['R'].copy_(emb.repeat(4,1));d['Token'].fill_(token_id)
 item=dict(token=token_id)
 for label,fn in [('vector',split),('wmma',full)]:
  fn();torch.cuda.synchronize();outdir=root/f'case-{token_id}-{label}';outdir.mkdir(exist_ok=True)
  try:check(d,outdir);passed=True
  except AssertionError:passed=False
  metrics=json.loads((outdir/'reference-comparison.json').read_text())
  item[label]=dict(passes_original_gate=passed,output=metrics['Out'])
  assert all(x['nonfinite']==0 for x in metrics.values())
  assert metrics['Ids']['exact']
 cases.append(item);(root/'embedding-cases.json').write_text(json.dumps(cases,indent=2))
d['R'].copy_(saved_r);d['Token'].fill_(42)
full();expected_output=d['Out'].clone()
for initial in [2147483648-args.grid,-args.grid]:
 d['Counter'].fill_(initial);full();torch.cuda.synchronize();assert d['Health'].item()==0
 torch.testing.assert_close(d['Out'],expected_output,rtol=0,atol=0)
cg=torch.cuda.CUDAGraph()
with torch.cuda.graph(cg):d['R'].neg_();full()
for _ in range(1000):cg.replay()
torch.cuda.synchronize();assert d['Health'].item()==0
actual=d['Out'].clone();full();torch.cuda.synchronize();torch.testing.assert_close(actual,d['Out'],rtol=0,atol=0)
for i in range(7):region(i)
torch.cuda.synchronize();torch.testing.assert_close(actual,d['Out'],rtol=0,atol=0)
d['R'].copy_(saved_r)
(root/'replay-validation.json').write_text(json.dumps(dict(replays=1000,changing_payload=True,counter_wrap=True,exact_against_regions=True,health=d['Health'].item()),indent=2))
# Same-session timing; report only run 2. No restoring copies in timed graphs.
spec=importlib.util.spec_from_file_location('vector_region',root.parent/'regions/region_kernel.py');vr=importlib.util.module_from_spec(spec);spec.loader.exec_module(vr)
def vector_region(i):return vr.region_kernel[(args.grid,)](*values,REGION=i,**kw)
def vector_full():return full_layer[(args.grid,)](*values,**kw)
def measure(fn,name):
 fn();torch.cuda.synchronize();gph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(gph):fn()
 for _ in range(10):gph.replay()
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  for _ in range(30):gph.replay()
  torch.cuda.synchronize()
 path=root/(name+'.json');prof.export_chrome_trace(str(path))
 ev=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel']
 n=len(ev)//30;assert n>0 and len(ev)==30*n
 times=[sum(e['dur'] for e in ev[i:i+n]) for i in range(0,len(ev),n)]
 assert d['Health'].item()==0
 return dict(median_us=statistics.median(times),samples_us=times,kernels=n)
results=[]
for run in [1,2]:
 row=dict(run=run,regions=[])
 for i,(a,b) in enumerate(regions):
  r=dict(region=i)
  for name,fn in [('split',lambda a=a,b=b:split(a,b)),('vector',lambda i=i:vector_region(i)),('wmma',lambda i=i:region(i))]:
   restore(fixtures[i]);r[name]=measure(fn,f'r{run}-region{i}-{name}')
  row['regions'].append(r);print('region timing',run,i,{k:v['median_us'] for k,v in r.items() if isinstance(v,dict)},flush=True)
 row['full']={name:measure(fn,f'r{run}-full-{name}') for name,fn in [('split',split),('vector',vector_full),('wmma',full)]}
 print('full timing',run,{k:v['median_us'] for k,v in row['full'].items()},flush=True)
 results.append(row);(root/'timings.json').write_text(json.dumps(results,indent=2))
print('ALL DONE; extra embedding gate results',cases,flush=True)
