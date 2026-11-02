from pathlib import Path
import torch,json,statistics
from mhc import mhc_region,mhc_project
from projection import region,project,region_tail,region_packed
root=Path(__file__).resolve().parent
f=torch.load(root/'mhc-fixture.pt',weights_only=True)
W,S,B,AN=[f[k].cuda().contiguous() for k in ['HAF','HAS','HAB','AN']]
R=f['inputs'][0].cuda().contiguous();kw=dict(num_warps=4,enable_fp_fusion=False)
def state():return [torch.zeros(24,device='cuda'),torch.zeros((128,24),device='cuda'),torch.zeros(4096,device='cuda',dtype=torch.bfloat16),torch.zeros(4096,device='cuda',dtype=torch.bfloat16),torch.zeros(1,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32)]
ref=state();mhc_region[(256,)](R,W,S,B,AN,*ref,**kw);torch.cuda.synchronize()
Wh=W.bfloat16();Wl=(W-Wh.float()).bfloat16();Wp=torch.cat([Wh,Wl],0).contiguous()
entries=[]
for bk,bn,grid in [(128,16,256)]+[(bk,bn,gr) for bk in [128,256,512] for bn in [16,32] for gr in sorted(set([256,((24+bn-1)//bn)*(16384//bk)]))]:
 st=state();name=f'bk{bk}_bn{bn}_g{grid}'
 def fn(st=st,bk=bk,bn=bn,grid=grid):return region_packed[(grid,)](R,Wp,S,B,AN,*st,BK=bk,BN=bn,GRID=grid,**kw)
 obj=fn();torch.cuda.synchronize();err=((st[3].float()-ref[3].float()).square().mean()/ref[3].float().square().mean()).sqrt().item()
 assert err<.001,(name,err)
 (root/(name+'.amdgcn')).write_text(obj.asm['amdgcn'])
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(128):fn()
 entries.append((name,st,g,err,dict(vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared)))
 print(name,err,flush=True)
rows=[]
for rnd in [1,2]:
 samples={n:[] for n,*_ in entries}
 for it in range(30):
  for n,st,g,err,res in entries[::1 if it%2==0 else -1]:
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/128)
 row={'round':rnd,'results':{n:dict(us=statistics.median(samples[n]),err=err,resources=res,health=st[-1].item()) for n,st,g,err,res in entries}}
 rows.append(row);(root/'packed-timings.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
