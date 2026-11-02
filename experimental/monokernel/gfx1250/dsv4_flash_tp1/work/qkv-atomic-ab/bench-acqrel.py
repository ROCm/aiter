import json,statistics,re,hashlib
from pathlib import Path
import torch
from safetensors import safe_open
from kernels import initialize,project,reduce_norm
root=Path(__file__).resolve().parent
torch.manual_seed(0)
assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1250')
p=Path('/data/DeepSeek-V4-Flash');idx=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
def get(name):
 with safe_open(str(p/idx[name]),framework='pt',device='cpu') as f:return f.get_tensor(name).cuda()
def w(name):return get('layers.0.'+name)
wa=w('attn.wq_a.weight');wk=w('attn.wkv.weight')
W=torch.cat([wa.view(torch.uint8),wk.view(torch.uint8)]).view(wa.dtype)
WS=torch.cat([w('attn.wq_a.scale').view(torch.uint8),w('attn.wkv.scale').view(torch.uint8)])
NW=torch.cat([w('attn.q_norm.weight').float(),w('attn.kv_norm.weight').float()])
# First case is an actual serving projection input, not only synthetic data.
live=torch.load('/work/live-vector/debug-live.pt',map_location='cpu')['vector']['X'].flatten().cuda()
inputs=[live]
emb=get('embed.weight')
an=w('attn_norm.weight').float()
for token in [0,1,7,42,43,100,1024,8192]:
 x=emb[token].float();inputs.append((x*torch.rsqrt(x.square().mean()+1.e-6)*an).bfloat16())
del emb
bank=[]
for x in inputs:
 z=x.float().reshape(-1,128);amax=z.abs().amax(-1).clamp_min(1.e-10)/448.
 e=(((amax.view(torch.int32)&0x7fffffff)+0x7fffff)>>23).clamp_max(254)
 scale=(e<<23).view(torch.float32)
 bank.append(((z/scale[:,None]).to(torch.float8_e4m3fn).flatten(),e.to(torch.uint8)))
X=bank[0][0].clone();XS=bank[0][1].clone()
P=torch.empty((8,1536),device='cuda',dtype=torch.float32)
Acc=torch.empty(1536,device='cuda',dtype=torch.float32)
SS=torch.empty(48,device='cuda',dtype=torch.float32)
Count=torch.empty(50,device='cuda',dtype=torch.int32)
Y=torch.empty(1536,device='cuda',dtype=torch.float32)
kw=dict(num_warps=4,enable_fp_fusion=False)
resources={}
def run(mode):
 if mode:initialize[(6,)](Acc,Count,**kw)
 obj=project[(384,)](X,XS,W,WS,NW,P,Acc,SS,Count,Y,MODE=mode,**kw)
 if mode in [0,2]:reduce_norm[(2,)](P,Acc,NW,Y,ATOMIC=mode==2,**kw)
 return obj
names=['A_scratch_separate_norm','B_atomic_fused_norm','C_atomic_separate_norm','D_scratch_fused_norm']
for mode,name in enumerate(names):
 obj=run(mode);torch.cuda.synchronize()
 isa=obj.asm['amdgcn'];(root/(name+'.amdgcn')).write_text(isa)
 resources[name]=dict(vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared,wmma=isa.count('v_wmma'),float_atomic_lines=[s.strip() for s in isa.splitlines() if 'atomic_add_f32' in s],scratch=[s.strip() for s in isa.splitlines() if '.amdhsa_private_segment_fixed_size' in s],sgpr_spills=re.findall(r'\.sgpr_spill_count:\s*(\d+)',isa),vgpr_spills=re.findall(r'\.vgpr_spill_count:\s*(\d+)',isa))
 assert 'v_wmma' in isa
 if mode in [1,2]:assert 'atomic_add_f32' in isa
print('resources',resources,flush=True)
(root/'resources.json').write_text(json.dumps(resources,indent=2))
wf=W.double()*torch.pow(2.,WS.double()-127).repeat_interleave(128,0).repeat_interleave(128,1)
def reference():
 xf=X.double()*torch.pow(2.,XS.double()-127).repeat_interleave(128)
 z=(wf@xf).bfloat16().float()
 return torch.cat([z[:1024]*torch.rsqrt(z[:1024].square().mean()+1.e-6)*NW[:1024],z[1024:]*torch.rsqrt(z[1024:].square().mean()+1.e-6)*NW[1024:]])
def metric(a,b):
 d=a.float()-b.float();return dict(relative_rms=(d.square().mean()/b.float().square().mean().clamp_min(1e-30)).sqrt().item(),max_abs=d.abs().max().item(),exact=torch.equal(a,b),finite=torch.isfinite(a).all().item())
checks=[]
for case,(xp,xs) in enumerate(bank):
 X.copy_(xp);XS.copy_(xs);ref=reference();run(0);base=Y.clone()
 for mode,name in enumerate(names):
  Acc.fill_(float('nan'));P.fill_(float('nan'));SS.fill_(float('nan'));Count.fill_(987);Y.fill_(float('nan'))
  run(mode);torch.cuda.synchronize()
  a=metric(Y,base);b=metric(Y,ref)
  checks.append(dict(case=case,variant=name,vs_A=a,vs_fp64_reference=b))
  assert a['finite'] and a['relative_rms']<.001 and b['relative_rms']<.001,checks[-1]
  if mode in [1,3]:assert torch.equal(Count[:48],torch.full_like(Count[:48],8)) and Count[48:].tolist()==[32,16]
(root/'correctness.json').write_text(json.dumps(checks,indent=2))
# Changing-input replay, poison already tested above. Reset is inside every graph.
stress=[]
for mode in [1,3]:
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):run(mode)
 worst=0.
 for i in range(1000):
  xp,xs=bank[i%len(bank)];X.copy_(xp);XS.copy_(xs);cg.replay()
  if i%100==0 or i==999:
   actual=Y.clone();run(0);m=metric(actual,Y);assert m['finite'] and m['relative_rms']<.001,m;worst=max(worst,m['relative_rms'])
 torch.cuda.synchronize();stress.append(dict(variant=names[mode],replays=1000,max_checked_rms_vs_A=worst))
(root/'stress.json').write_text(json.dumps(stress,indent=2))
X.copy_(bank[0][0]);XS.copy_(bank[0][1]);torch.cuda.synchronize()
# Average graph elapsed time per invocation, including reset and launch gaps.
repeat=100;graphs=[]
for mode in range(4):
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):
  for _ in range(repeat):run(mode)
 graphs.append(cg)
for cg in graphs:
 for _ in range(10):cg.replay()
torch.cuda.synchronize()
rounds=[]
for rnd in [1,2]:
 samples={n:[] for n in names}
 for sample in range(30):
  order=range(4) if sample%2==0 else reversed(range(4))
  for mode in order:
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   start.record();graphs[mode].replay();end.record();end.synchronize()
   samples[names[mode]].append(start.elapsed_time(end)*1000/repeat)
 rows={n:dict(median_us=statistics.median(v),min_us=min(v),max_us=max(v),samples_us=v) for n,v in samples.items()}
 rounds.append(dict(round=rnd,graph_repeats=repeat,results=rows));print('ROUND',rnd,{n:r['median_us'] for n,r in rows.items()},flush=True)
(root/'timings.json').write_text(json.dumps(rounds,indent=2))
# Per-launch costs from a separate profiler pass, not substituted for graph timings.
for mode,name in enumerate(names):
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):run(mode)
 for _ in range(10):cg.replay()
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  for _ in range(30):cg.replay()
  torch.cuda.synchronize()
 prof.export_chrome_trace(str(root/(name+'-trace.json')))
(root/'manifest.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0),arch=torch.cuda.get_device_properties(0).gcnArchName,torch=torch.__version__,source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}),indent=2))
print('DONE',flush=True)
