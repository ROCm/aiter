import torch,aiter,json
from post import post
torch.manual_seed(901);results={}
for fma in [2]:
 mx=0;neq=0
 for seed in range(64):
  torch.manual_seed(seed+9300);r=torch.randn((1,4,4096),device='cuda',dtype=torch.bfloat16);x=torch.randn((1,4096),device='cuda',dtype=torch.bfloat16);m=torch.rand(24,device='cuda');m[4:8]*=2
  ref=torch.empty_like(r);out=torch.empty_like(r)
  aiter.mhc_post(ref,x,r,m[4:8].view(1,4,1),m[8:].view(1,4,4))
  obj=post[(256,)](r,x,m,out,FMA=fma,num_warps=4,enable_fp_fusion=False)
  torch.cuda.synchronize();mx=max(mx,float((out.float()-ref.float()).abs().max()));neq+=int((out!=ref).sum())
  if bool((out!=ref).any()):
   ids=(out!=ref).nonzero();print("DIFF",seed,ids.tolist(),out[out!=ref].tolist(),ref[out!=ref].tolist(),flush=True);torch.save(dict(r=r,x=x,m=m,ref=ref,out=out),"diff.pt")
 results[str(fma)]={'max_abs':mx,'unequal':neq,'vgpr':obj.n_regs,'spills':obj.n_spills}
 print(fma,results[str(fma)],flush=True)
json.dump(results,open('check2.json','w'),indent=2)
