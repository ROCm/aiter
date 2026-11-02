import sys,json,statistics
from pathlib import Path
sys.path[:0]=['/app/aiter','/app/ATOM','/work/atom-actual-microbench/attention']
import torch,aiter
from graphbatch import FlatGraphBatch
from kernel import combine_post
root=Path(__file__).resolve().parent
M=64;torch.manual_seed(67890)
r=torch.randn(M,4,4096,device='cuda').bfloat16();rsh=aiter.mhc_res_shuffle(r)
route=torch.randn(M,4096,device='cuda').bfloat16();shared=torch.randn_like(route)
p=torch.rand(M,4,1,device='cuda');c=torch.rand(M,4,4,device='cuda');out=torch.empty_like(r);ref=torch.empty_like(r)
def native():
 x=route+shared
 aiter.mhc_post(ref,x,rsh,p,c,res_preshuffle=True)
 return ref
def fused():
 global obj
 obj=combine_post[(M*32,)](route,shared,rsh,p,c,out,num_warps=4,enable_fp_fusion=False)
 return out
checks=[]
for seed in range(16):
 torch.manual_seed(67890+seed);route.copy_(torch.randn_like(route));shared.copy_(torch.randn_like(shared));r.copy_(torch.randn_like(r));rsh.copy_(aiter.mhc_res_shuffle(r))
 native();fused();torch.cuda.synchronize();checks.append({'seed':seed,'equal':torch.equal(out,ref),'max_abs':float((out.float()-ref.float()).abs().max()),'rms':float(((out.float()-ref.float()).square().mean()/ref.float().square().mean()).sqrt())})
json.dump(checks,open(root/'validation.json','w'),indent=2);print(checks,flush=True)
graphs={n:FlatGraphBatch(fn,128) for n,fn in [('native',native),('fused',fused)]};rounds=[]
for rnd in [1,2]:
 for g in graphs.values():
  for _ in range(5):g.replay()
 raw={n:[] for n in graphs}
 for i in range(40):
  for n in list(graphs)[::1 if i%2==0 else -1]:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();graphs[n].replay();b.record();b.synchronize();raw[n].append(a.elapsed_time(b)*1000/128)
 rounds.append({'round':rnd,'median_us':{n:statistics.median(x) for n,x in raw.items()},'samples_us':raw})
json.dump(rounds,open(root/'timings.json','w'),indent=2);print(rounds[-1]['median_us'],flush=True)
json.dump({'vgpr':obj.n_regs,'lds':obj.metadata.shared,'spills':obj.n_spills,'graph_nodes':{n:g.kernel_count for n,g in graphs.items()}},open(root/'resources.json','w'),indent=2)
(root/'kernel.amdgcn').write_text(obj.asm['amdgcn'])
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 native();fused();torch.cuda.synchronize()
prof.export_chrome_trace(str(root/'trace.json'))
