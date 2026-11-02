"""Isolated checkpoint-backed full-layer test, not an ATOM serving benchmark."""
import argparse, inspect, json, time
from pathlib import Path
import torch
from safetensors import safe_open
from kernel import full_layer

root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--compile-only',action='store_true');ap.add_argument('--grid',type=int,default=64);args=ap.parse_args()
torch.manual_seed(0)
assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1250')
assert args.grid<=torch.cuda.get_device_properties(0).multi_processor_count
p=Path('/data/DeepSeek-V4-Flash');index=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
handles={}
def tensor(key):
 name=index[key]
 if name not in handles:handles[name]=safe_open(str(p/name),framework='pt',device='cpu')
 return handles[name].get_tensor(key)
def load(key):return tensor('layers.0.'+key).cuda()
def scale(key):return load(key).view(torch.uint8)
d={}
for short,key in [('HAF','hc_attn_fn'),('HAS','hc_attn_scale'),('HAB','hc_attn_base'),('HFN','hc_ffn_fn'),('HFS','hc_ffn_scale'),('HFB','hc_ffn_base'),('AN','attn_norm.weight'),('FN','ffn_norm.weight'),('QN','attn.q_norm.weight'),('KN','attn.kv_norm.weight'),('Sink','attn.attn_sink'),('Gate','ffn.gate.weight'),('Tid','ffn.gate.tid2eid')]:d[short]=load(key)
for w,s,key in [('QA','QAS','attn.wq_a'),('QB','QBS','attn.wq_b'),('KVW','KVS','attn.wkv'),('OA','OAS','attn.wo_a'),('OB','OBS','attn.wo_b'),('SW1','SS1','ffn.shared_experts.w1'),('SW3','SS3','ffn.shared_experts.w3'),('SW2','SS2','ffn.shared_experts.w2')]:
 d[w]=load(key+'.weight');d[s]=scale(key+'.scale')
print('Dense weights loaded',flush=True)
for w,s,suffix in [('EW1','ES1','w1'),('EW3','ES3','w3'),('EW2','ES2','w2')]:
 ws=load('ffn.experts.0.'+suffix+'.weight');ss=scale('ffn.experts.0.'+suffix+'.scale')
 d[w]=torch.empty((256,*ws.shape),dtype=ws.dtype,device='cuda');d[s]=torch.empty((256,*ss.shape),dtype=ss.dtype,device='cuda')
 for e in range(256):
  d[w][e].copy_(tensor(f'layers.0.ffn.experts.{e}.{suffix}.weight'))
  d[s][e].copy_(tensor(f'layers.0.ffn.experts.{e}.{suffix}.scale').view(torch.uint8))
 print(w,'loaded',flush=True)
token=42
with safe_open(str(p/index['embed.weight']),framework='pt',device='cpu') as f:embed=f.get_slice('embed.weight')[token:token+1].cuda()
d['R']=embed.repeat(4,1).contiguous();d['Out']=torch.empty_like(d['R']);d['Token']=torch.tensor([token],device='cuda',dtype=torch.int64)
d['Cache']=torch.randn((128,512),device='cuda',dtype=torch.bfloat16)*.2
# Controlled rotary angle; uses the same exact input table in both paths. Live
# ATOM replay must supply its YaRN table instead of this unit-test fixture.
angles=1024*10000.**(-torch.arange(32,device='cuda',dtype=torch.float32)/32)
d['Cos']=angles.cos();d['Sin']=angles.sin()
sizes={'MixA':24,'MixF':24,'R2':16384,'X':4096,'XQ':4096,'QL':1024,'QLQ':1024,'Q':32768,'KV':512,'QRot':32768,'KVRot':512,'AO':32768,'AR':32768,'AQ':32768,'OL':8192,'OLQ':8192,'Attn':4096,'FF':4096,'FFQ':4096,'FFMX':4096,'SG':2048,'SU':2048,'SA':2048,'SAQ':2048,'SO':4096,'Logits':256,'Ids':6,'Gamma':6,'EG':12288,'EU':12288,'EA':12288,'EAQ':12288,'EO':24576,'Sum':4096}
fp32={'QL','Q','KV','EG','EU','EA','MixA','MixF','XQ','QLQ','AQ','OLQ','FFQ','FFMX','SAQ','EAQ','Gamma'}
for k,n in sizes.items():d[k]=torch.empty(n,device='cuda',dtype=torch.int32 if k=='Ids' else torch.float32 if k in fp32 else torch.bfloat16)
d['Counter']=torch.zeros(1,device='cuda',dtype=torch.int32);d['Health']=torch.zeros_like(d['Counter'])
keys=list(inspect.signature(full_layer.fn).parameters)[:-1]; values=[d[k] for k in keys]
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
