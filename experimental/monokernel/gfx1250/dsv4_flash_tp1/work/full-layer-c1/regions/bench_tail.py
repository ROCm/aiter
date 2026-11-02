from split_control import split_stage, NUM_STAGES
from region_kernel import region_kernel
import statistics
mapping=json.loads((root/'mapping.json').read_text())
kw=dict(GRID=args.grid,num_warps=4,enable_fp_fusion=False)
def stage(i):split_stage[(args.grid,)](*values,STAGE=i,**kw)
def region(i):return region_kernel[(args.grid,)](*values,REGION=i,**kw)
def split():
 for i in range(NUM_STAGES):stage(i)
split();torch.cuda.synchronize()
expected={k:d[k].clone() for k in [*sizes,'Out','Cache']}
resources=[]
for i,m in enumerate(mapping):
 c=region(i);torch.cuda.synchronize()
 (root/(m['name']+'.amdgcn')).write_text(c.asm['amdgcn'])
 resources.append(dict(name=m['name'],n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared))
# Run all regions in order from the original layer input, compare all intermediates.
for i in range(len(mapping)):region(i)
torch.cuda.synchronize()
checks={k:torch.equal(d[k],v) for k,v in expected.items()}
assert all(checks.values()),checks
assert d['Health'].item()==0
(root/'correctness.json').write_text(json.dumps(checks,indent=2))
(root/'resources.json').write_text(json.dumps(resources,indent=2))
# Restore each region's canonical inputs before each timed replay, outside event range.
# This is necessary because normalization stages overwrite their own input.
split();torch.cuda.synchronize()
snap={k:d[k].clone() for k in [*sizes,'Out','Cache']}
def restore():
 for k,v in snap.items():d[k].copy_(v)
# Obtain exact pre-stage fixtures by walking once from beginning.
fixtures=[]
for j in range(NUM_STAGES):
 fixtures.append({k:d[k].clone() for k in [*sizes,'Out','Cache']})
 stage(j)
torch.cuda.synchronize()
def measure(fn,fixture):
 def reset():
  for k,v in fixture.items():d[k].copy_(v)
 reset();fn();torch.cuda.synchronize()
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):
  reset();fn()
 for _ in range(5):cg.replay()
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  for _ in range(30):cg.replay()
  torch.cuda.synchronize()
 trace_path=root/f'measure-{measure.count:03d}.json'
 measure.count+=1
 prof.export_chrome_trace(str(trace_path))
 trace=json.loads(trace_path.read_text())
 events=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and (e['name'].startswith('split_stage') or e['name'].startswith('region_kernel'))]
 assert len(events)%30==0 and len(events)>0,(len(events),trace_path)
 n=len(events)//30
 samples=[sum(e['dur'] for e in events[j:j+n]) for j in range(0,len(events),n)]
 assert d['Health'].item()==0
 return dict(median_us=statistics.median(samples),mean_us=statistics.mean(samples),samples_us=samples,trace=str(trace_path.name),kernels_per_replay=n)
measure.count=0
rounds=[]
for run in [1,2]:
 st=[measure(lambda j=j:stage(j),fixtures[j]) for j in range(NUM_STAGES)]
 rows=[]
 for i,m in enumerate(mapping):
  fused=measure(lambda i=i:region(i),fixtures[m['start']])
  def group(m=m):
   for j in range(m['start'],m['end']+1):stage(j)
  grouped=measure(group,fixtures[m['start']])
  row=dict(name=m['name'],start=m['start'],end=m['end'],unfused_sum_us=sum(x['median_us'] for x in st[m['start']:m['end']+1]),fused_us=fused['median_us'],unfused_group_us=grouped['median_us'],fused=fused,grouped=grouped)
  rows.append(row);print('round',run,{k:v for k,v in row.items() if k not in ['fused','grouped']},flush=True)
 rounds.append(dict(run=run,stages=st,regions=rows))
 (root/'timings.json').write_text(json.dumps(rounds,indent=2))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 for i in range(len(mapping)):region(i)
 torch.cuda.synchronize()
prof.export_chrome_trace(str(root/'regions-trace.json'))
print('DONE',flush=True)
