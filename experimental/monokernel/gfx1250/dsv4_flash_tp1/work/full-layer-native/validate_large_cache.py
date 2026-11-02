from layer_setup import *
from serving_kernel import serving_full_layer
f=LayerState();real=torch.empty((190000000,512),device='cuda',dtype=torch.bfloat16);indices=torch.arange(189608625,189608753,device='cuda',dtype=torch.int32);ptr=torch.tensor([0,128],device='cuda',dtype=torch.int32);dest=torch.tensor([189608752],device='cuda',dtype=torch.int32);pos=torch.tensor([1024],device='cuda',dtype=torch.int64)
rows=[]
for position in [1024,1536,2047]:
 f.a.setpos(position);pos.fill_(position);real[189608625:189608753].copy_(f.a.s.attn.cache)
 ref=f.fused().clone();torch.cuda.synchronize()
 obj=serving_full_layer[(256,)](f.attn_args(),f.f.args(),pos,f.a.nq.cos,f.a.nq.sin,real,indices,ptr,dest,**f.f.meta(),REALPAGES=190000000,num_warps=4,enable_fp_fusion=False)
 torch.cuda.synchronize();rows.append({'position':position,'equal':torch.equal(ref,aiter.mhc_res_unshuffle(f.f.out)),'cache_equal':torch.equal(real[189608752],f.a.s.kv),'health':f.health(),'vgpr':obj.n_regs,'lds':obj.metadata.shared,'spills':obj.n_spills})
(root/'adapter-large-cache-validation.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
(root/'serving.amdgcn').write_text(obj.asm['amdgcn'])
