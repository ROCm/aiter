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

split(0,22);before=snapshot();split(23,30);expected=snapshot()
for bn in [16,32,64,128]:
 restore(before)
 c=region_kernel[(args.grid,)](*values,REGION=5,RBN=bn,**kw,**extra);torch.cuda.synchronize()
 checkvals=compare(expected);res=audit(c,'tune-moe-'+str(bn))
 assert max(v['relative_rms'] for v in checkvals.values())<.005
 for run in [1,2]:
  restore(before)
  t=measure(lambda:region_kernel[(args.grid,)](*values,REGION=5,RBN=bn,**kw,**extra),f'tune-moe-{bn}-r{run}')
  result=dict(bn=bn,run=run,timing=t,resources=res)
  print(result,flush=True)
  (root/f'tune-moe-{bn}-r{run}-result.json').write_text(json.dumps(result,indent=2))
